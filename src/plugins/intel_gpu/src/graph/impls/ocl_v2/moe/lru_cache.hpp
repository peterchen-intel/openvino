// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cldnn {
struct event;
}  // namespace cldnn

namespace ov::intel_gpu::ocl::moe {

/// Opaque completion handle. The cache only stores it and hands it back, it never dereferences it, so the slot state
/// machine has no dependency on the runtime.
using EventHandle = std::shared_ptr<cldnn::event>;

/// Lifecycle of a resident slot as seen by the lookahead scheduler.
///   READY         the content is valid and kernels may read it
///   PENDING_LOAD  an asynchronous load is in flight, the content is not valid yet
/// "Pinned" is an orthogonal flag: a slot referenced by the layer being computed must neither be evicted nor reused,
/// no matter whether its load is still in flight.
enum class SlotStatus : uint8_t {
    READY,
    PENDING_LOAD,
};

struct ExpertSlot {
    uint32_t expert = 0;
    size_t slot = 0;
    EventHandle ready_event;  // F: event of the load that is already in flight
    EventHandle war_event;    // M and speculative: last reader of the evicted content, the new load must wait for it
};

/// Result of partition_3way. The sets are pairwise disjoint and every unique requested expert is in exactly one.
struct ExpertPartition {
    std::vector<ExpertSlot> ready;     // R: resident and valid
    std::vector<ExpertSlot> inflight;  // F: a load is already in flight (issued by an earlier prefetch)
    std::vector<ExpertSlot> missing;   // M: not resident, a slot is reserved, the caller must start the load
};

class LRUCache {
public:
    explicit LRUCache(size_t max_total_experts);

    std::pair<size_t, bool> get_lru_item(size_t expert);

    void evict_one();

    size_t size() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_total_experts;
    }

    void set_filled(size_t lru_expert_no) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (lru_expert_no >= m_filled_list.size()) {
            return;
        }
        m_filled_list[lru_expert_no] = true;
    }

    bool is_initialized() const {
        return m_initialized;
    }
    void set_initialized() {
        m_initialized = true;
    }

    // ---------------------------------------------------------------------------------------------------------
    // Lookahead scheduling extension. Everything below shares the mutex with the legacy methods, so partitioning,
    // pinning and speculative reservation are atomic with respect to each other: an expert never gets two loads.
    // The legacy synchronous paths must call wait_no_pending() before get_lru_item(): they would otherwise treat a
    // slot that is still being written by an asynchronous load as a miss and load it a second time.
    // ---------------------------------------------------------------------------------------------------------

    /// Splits `experts` (duplicates allowed) into R/F/M, pins every involved slot and reserves a slot for each M.
    /// Returns nullopt, without any state change, when the unique experts cannot all be resident at once.
    std::optional<ExpertPartition> partition_3way(const std::vector<uint32_t>& experts);

    /// Attaches the completion event of the load of a reserved slot, so later partition_3way calls report it as F.
    void set_ready_event(size_t slot, EventHandle event);

    /// Reserves a slot for a speculative load. nullopt if the expert is already resident or in flight, if
    /// `max_pending` loads are already in flight, or if no unpinned READY slot can be evicted. The new entry is the
    /// first eviction candidate, so a wrong prediction is the first thing to be recycled.
    std::optional<ExpertSlot> reserve_speculative(uint32_t expert, size_t max_pending);

    /// PENDING_LOAD -> READY. Wakes up wait_no_pending().
    void mark_ready(size_t slot);

    /// Drops a reservation whose load failed: the expert becomes not resident and the slot is free again.
    void abort_pending(size_t slot);

    /// Releases the pin and records the event of the last kernel reading each slot (WAR protection).
    void unpin(const std::vector<size_t>& slots, const EventHandle& use_event);

    /// Blocks until no load is in flight.
    void wait_no_pending();

    /// While disabled, reserve_speculative() refuses. A path that is not aware of in-flight loads (the legacy
    /// synchronous one) disables speculation and then calls wait_no_pending(); afterwards nothing can write a slot
    /// behind its back. The lookahead path enables it again.
    void set_speculation_enabled(bool enabled);

    size_t pending_count() const;
    size_t pinned_count() const;
    SlotStatus status_of_slot(size_t slot) const;
    bool is_pinned(size_t slot) const;
    std::optional<size_t> find_slot(uint32_t expert) const;

private:
    bool m_initialized = false;
    struct Key {
        size_t expert;
        bool operator==(const Key& other) const noexcept {
            return expert == other.expert;
        }
    };

    struct KeyHash {
        std::size_t operator()(const Key& k) const noexcept {
            return std::hash<size_t>()(k.expert);
        }
    };

    struct Node {
        size_t expert;
        size_t lru_expert_no;
    };

    struct SlotMeta {
        SlotStatus status = SlotStatus::READY;
        uint32_t pin_count = 0;
        EventHandle ready_event;
        EventHandle last_use_event;
    };

    size_t m_max_total_experts;
    size_t m_total_experts;

    std::list<Node> m_list;  // front = first eviction candidate
    std::vector<bool> m_filled_list;
    std::vector<SlotMeta> m_meta;
    std::vector<size_t> m_free_slots;  // stack, the next slot to hand out is at the back
    size_t m_pending = 0;
    bool m_speculation_enabled = true;
    std::unordered_map<Key, std::list<Node>::iterator, KeyHash> m_map;
    mutable std::mutex m_mutex;
    std::condition_variable m_pending_cv;

    void move_to_end(std::list<Node>::iterator it);
    bool is_evictable(const Node& node) const;
    std::list<Node>::iterator find_victim();
    void release_node(std::list<Node>::iterator it);
    size_t evictable_count() const;
    // Returns a free slot, evicting the oldest evictable entry if needed. The caller guarantees one exists.
    size_t take_slot(EventHandle* war_event);
    size_t insert_node(size_t expert, bool at_front, EventHandle* war_event);
};

}  // namespace ov::intel_gpu::ocl::moe
