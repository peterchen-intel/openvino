// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lru_cache.hpp"

#include <algorithm>
#include <unordered_set>

namespace ov::intel_gpu::ocl::moe {

LRUCache::LRUCache(size_t max_total_experts) : m_max_total_experts(max_total_experts), m_total_experts(0) {
    m_filled_list.resize(max_total_experts, false);
    m_meta.resize(max_total_experts);
    // The stack is popped from the back, so slots are handed out as 0, 1, 2, ...
    m_free_slots.reserve(max_total_experts);
    for (size_t i = max_total_experts; i > 0; --i) {
        m_free_slots.push_back(i - 1);
    }
}

void LRUCache::move_to_end(std::list<Node>::iterator it) {
    if (std::next(it) == m_list.end()) {
        return;
    }
    m_list.splice(m_list.end(), m_list, it);
}

bool LRUCache::is_evictable(const Node& node) const {
    const auto& meta = m_meta[node.lru_expert_no];
    return meta.pin_count == 0 && meta.status == SlotStatus::READY;
}

std::list<LRUCache::Node>::iterator LRUCache::find_victim() {
    return std::find_if(m_list.begin(), m_list.end(), [this](const Node& n) {
        return is_evictable(n);
    });
}

void LRUCache::release_node(std::list<Node>::iterator it) {
    const size_t slot = it->lru_expert_no;
    auto& meta = m_meta[slot];
    if (meta.status == SlotStatus::PENDING_LOAD) {
        --m_pending;
    }
    meta.status = SlotStatus::READY;
    meta.pin_count = 0;
    meta.ready_event.reset();
    // last_use_event is kept on purpose: it is the WAR dependency of the next load into this slot.
    m_filled_list[slot] = false;
    m_map.erase(Key{it->expert});
    m_list.erase(it);
    m_free_slots.push_back(slot);
    --m_total_experts;
}

size_t LRUCache::take_slot(EventHandle* war_event) {
    if (m_free_slots.empty()) {
        auto victim = find_victim();
        if (victim == m_list.end()) {
            // Legacy behaviour for callers that do not use pinning: recycle the oldest entry unconditionally.
            victim = m_list.begin();
        }
        release_node(victim);
    }
    const size_t slot = m_free_slots.back();
    m_free_slots.pop_back();
    if (war_event) {
        *war_event = std::move(m_meta[slot].last_use_event);
    }
    m_meta[slot].last_use_event.reset();
    return slot;
}

size_t LRUCache::insert_node(size_t expert, bool at_front, EventHandle* war_event) {
    const size_t slot = take_slot(war_event);
    if (at_front) {
        m_list.push_front(Node{expert, slot});
        m_map[Key{expert}] = m_list.begin();
    } else {
        m_list.push_back(Node{expert, slot});
        m_map[Key{expert}] = std::prev(m_list.end());
    }
    ++m_total_experts;
    return slot;
}

void LRUCache::evict_one() {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto victim = find_victim();
    if (victim != m_list.end()) {
        release_node(victim);
    }
}

std::pair<size_t, bool> LRUCache::get_lru_item(size_t expert) {
    std::lock_guard<std::mutex> lock(m_mutex);

    auto it = m_map.find(Key{expert});
    if (it == m_map.end()) {
        const size_t slot = insert_node(expert, false, nullptr);
        return {slot, false};
    }
    move_to_end(it->second);
    const size_t slot = it->second->lru_expert_no;
    return {slot, m_filled_list[slot]};
}

std::optional<ExpertPartition> LRUCache::partition_3way(const std::vector<uint32_t>& experts) {
    std::unique_lock<std::mutex> lock(m_mutex);

    std::vector<uint32_t> unique;
    unique.reserve(experts.size());
    {
        std::unordered_set<uint32_t> seen;
        for (auto e : experts) {
            if (seen.insert(e).second) {
                unique.push_back(e);
            }
        }
    }
    if (unique.size() > m_max_total_experts) {
        return std::nullopt;
    }

    // A speculative slot is PENDING without an event only between reserve_speculative() and the moment its worker has
    // issued the load. Reporting it as F without an event would leave the caller nothing to wait for.
    m_pending_cv.wait(lock, [&] {
        return std::none_of(unique.begin(), unique.end(), [&](uint32_t e) {
            auto it = m_map.find(Key{e});
            if (it == m_map.end()) {
                return false;
            }
            const auto& meta = m_meta[it->second->lru_expert_no];
            return meta.status == SlotStatus::PENDING_LOAD && !meta.ready_event;
        });
    });

    // Classify and pin the resident ones. Pinning first makes them invisible to the victim search below.
    ExpertPartition result;
    std::vector<uint32_t> to_load;                   // not resident
    std::vector<std::pair<uint32_t, size_t>> stale;  // resident entry without valid content: reload into the same slot
    std::vector<size_t> pinned_here;
    for (auto e : unique) {
        auto it = m_map.find(Key{e});
        if (it == m_map.end()) {
            to_load.push_back(e);
            continue;
        }
        const size_t slot = it->second->lru_expert_no;
        auto& meta = m_meta[slot];
        ++meta.pin_count;
        pinned_here.push_back(slot);
        if (meta.status == SlotStatus::PENDING_LOAD) {
            result.inflight.push_back({e, slot, meta.ready_event, nullptr});
        } else if (m_filled_list[slot]) {
            result.ready.push_back({e, slot, nullptr, nullptr});
        } else {
            stale.emplace_back(e, slot);
        }
    }

    size_t evictable = 0;
    for (const auto& node : m_list) {
        evictable += is_evictable(node) ? 1 : 0;
    }
    if (m_free_slots.size() + evictable < to_load.size()) {
        for (auto slot : pinned_here) {
            --m_meta[slot].pin_count;
        }
        return std::nullopt;
    }

    for (auto e : to_load) {
        EventHandle war;
        const size_t slot = insert_node(e, false, &war);
        auto& meta = m_meta[slot];
        meta.status = SlotStatus::PENDING_LOAD;
        meta.pin_count = 1;
        meta.ready_event.reset();
        ++m_pending;
        result.missing.push_back({e, slot, nullptr, std::move(war)});
    }
    for (const auto& [e, slot] : stale) {
        auto& meta = m_meta[slot];
        meta.status = SlotStatus::PENDING_LOAD;
        meta.ready_event.reset();
        ++m_pending;
        result.missing.push_back({e, slot, nullptr, std::move(meta.last_use_event)});
        meta.last_use_event.reset();
    }

    // Refresh the recency of everything the current layer uses.
    for (const auto* group : {&result.ready, &result.inflight}) {
        for (const auto& s : *group) {
            move_to_end(m_map.at(Key{s.expert}));
        }
    }
    return result;
}

void LRUCache::set_ready_event(size_t slot, EventHandle event) {
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (slot >= m_meta.size()) {
            return;
        }
        m_meta[slot].ready_event = std::move(event);
    }
    m_pending_cv.notify_all();
}

std::optional<ExpertSlot> LRUCache::reserve_speculative(uint32_t expert, size_t max_pending) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_map.find(Key{expert}) != m_map.end()) {
        return std::nullopt;
    }
    if (!m_speculation_enabled || m_pending >= max_pending) {
        return std::nullopt;
    }
    if (m_free_slots.empty() && find_victim() == m_list.end()) {
        return std::nullopt;
    }
    EventHandle war;
    const size_t slot = insert_node(expert, true, &war);
    auto& meta = m_meta[slot];
    meta.status = SlotStatus::PENDING_LOAD;
    meta.pin_count = 0;
    meta.ready_event.reset();
    ++m_pending;
    return ExpertSlot{expert, slot, nullptr, std::move(war)};
}

void LRUCache::mark_ready(size_t slot) {
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (slot >= m_meta.size() || m_meta[slot].status != SlotStatus::PENDING_LOAD) {
            return;
        }
        m_meta[slot].status = SlotStatus::READY;
        m_meta[slot].ready_event.reset();
        m_filled_list[slot] = true;
        --m_pending;
    }
    m_pending_cv.notify_all();
}

void LRUCache::abort_pending(size_t slot) {
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (slot >= m_meta.size() || m_meta[slot].status != SlotStatus::PENDING_LOAD) {
            return;
        }
        auto& meta = m_meta[slot];
        if (meta.pin_count > 0) {
            // A kernel of the current layer already depends on this slot. Keep the entry (so the pin stays valid) but
            // mark the content as invalid: the next partition_3way reloads it in place.
            meta.status = SlotStatus::READY;
            meta.ready_event.reset();
            m_filled_list[slot] = false;
            --m_pending;
        } else {
            auto it = std::find_if(m_list.begin(), m_list.end(), [slot](const Node& n) {
                return n.lru_expert_no == slot;
            });
            if (it != m_list.end()) {
                release_node(it);
            }
        }
    }
    m_pending_cv.notify_all();
}

void LRUCache::unpin(const std::vector<size_t>& slots, const EventHandle& use_event) {
    std::lock_guard<std::mutex> lock(m_mutex);
    for (auto slot : slots) {
        if (slot >= m_meta.size()) {
            continue;
        }
        auto& meta = m_meta[slot];
        if (meta.pin_count > 0) {
            --meta.pin_count;
        }
        if (use_event) {
            meta.last_use_event = use_event;
        }
    }
}

void LRUCache::wait_no_pending() {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_pending_cv.wait(lock, [this] {
        return m_pending == 0;
    });
}

void LRUCache::set_speculation_enabled(bool enabled) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_speculation_enabled = enabled;
}

size_t LRUCache::pending_count() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_pending;
}

size_t LRUCache::pinned_count() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return static_cast<size_t>(std::count_if(m_meta.begin(), m_meta.end(), [](const SlotMeta& m) {
        return m.pin_count > 0;
    }));
}

SlotStatus LRUCache::status_of_slot(size_t slot) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_meta.at(slot).status;
}

bool LRUCache::is_pinned(size_t slot) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_meta.at(slot).pin_count > 0;
}

std::optional<size_t> LRUCache::find_slot(uint32_t expert) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_map.find(Key{expert});
    if (it == m_map.end()) {
        return std::nullopt;
    }
    return it->second->lru_expert_no;
}

}  // namespace ov::intel_gpu::ocl::moe
