// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <map>
#include <mutex>
#include <set>
#include <thread>
#include <vector>
#include <random>
#include <optional>

#include "ocl_v2/moe/residency/expert_prefetch_scheduler.hpp"
#include "test_utils/test_utils.h"

using namespace ov::intel_gpu::ocl::moe;
using namespace ov::intel_gpu::ocl::moe::lookahead;

namespace {

// Event stand-in: LRUCache and the scheduler only pass handles around, the loader below is the only one that looks inside.
struct FakeEvent {
    std::atomic<bool> signalled{false};
};

EventHandle as_handle(const std::shared_ptr<FakeEvent>& e) {
    return EventHandle(e, reinterpret_cast<cldnn::event*>(e.get()));
}
FakeEvent* fake(const EventHandle& h) {
    return reinterpret_cast<FakeEvent*>(h.get());
}

class FakeLoader : public IExpertLoader {
public:
    struct Load {
        uint32_t expert;
        size_t layer;
        size_t slot;
        size_t chunk;
        EventHandle war;
    };

    EventHandle make_ready_event() override {
        return as_handle(std::make_shared<FakeEvent>());
    }
    void signal(const EventHandle& e) noexcept override {
        fake(e)->signalled = true;
        std::lock_guard<std::mutex> lock(m_mutex);
        ++m_signals;
    }
    Completion begin_load(const PrefetchTarget& target, uint32_t expert, size_t slot, const EventHandle& war, size_t chunk) override {
        std::unique_lock<std::mutex> lock(m_mutex);
        if (m_fail_expert && *m_fail_expert == expert) {
            throw std::runtime_error("injected load failure");
        }
        m_order.push_back(Load{expert, target.layer_index, slot, chunk, war});
        m_in_use_chunks.insert(chunk);
        m_max_chunks = std::max(m_max_chunks, m_in_use_chunks.size());
        EXPECT_EQ(m_chunk_users[chunk], 0) << "staging chunk " << chunk << " reused before its copy finished";
        ++m_chunk_users[chunk];
        m_cv.notify_all();
        return [this, chunk, expert] {
            std::unique_lock<std::mutex> lk(m_mutex);
            m_cv.wait(lk, [&] {
                return m_gate_open || m_released.count(expert) > 0;
            });
            --m_chunk_users[chunk];
            m_in_use_chunks.erase(chunk);
        };
    }

    void hold_completions() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_gate_open = false;
    }
    void open_gate() {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_gate_open = true;
        }
        m_cv.notify_all();
    }
    void fail_on(uint32_t expert) {
        m_fail_expert = expert;
    }
    // Polls instead of condition_variable::wait_for: libstdc++ 11 implements wait_for with pthread_cond_clockwait, which
    // ThreadSanitizer 11 does not intercept and reports as a false double lock / data race.
    bool wait_started(size_t n, int ms = 3000) {
        const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
        while (std::chrono::steady_clock::now() < end) {
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                if (m_order.size() >= n) {
                    return true;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_order.size() >= n;
    }
    std::vector<Load> order() {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_order;
    }
    size_t max_chunks() {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_max_chunks;
    }

private:
    std::mutex m_mutex;
    std::condition_variable m_cv;
    bool m_gate_open = true;
    std::set<uint32_t> m_released;
    std::vector<Load> m_order;
    std::set<size_t> m_in_use_chunks;
    std::map<size_t, int> m_chunk_users;
    size_t m_max_chunks = 0;
    size_t m_signals = 0;
    std::optional<uint32_t> m_fail_expert;
};

PrefetchTarget make_target(size_t layer, size_t cap) {
    return PrefetchTarget{std::make_shared<LRUCache>(cap), nullptr, layer};
}

bool wait_for(const std::function<bool()>& pred, int ms = 3000) {
    auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
    while (std::chrono::steady_clock::now() < end) {
        if (pred()) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return pred();
}

}  // namespace

TEST(moe_lookahead_scheduler, demand_load_signals_event_and_marks_ready) {
    auto loader = std::make_shared<FakeLoader>();
    PrefetchScheduler sched(loader, 2);
    auto t = make_target(0, 4);
    auto part = t.cache->partition_3way({1, 2});
    ASSERT_TRUE(part.has_value());
    auto events = sched.submit_demand(t, part->missing);
    ASSERT_EQ(events.size(), 2U);
    sched.quiesce();
    for (auto& e : events) {
        EXPECT_TRUE(fake(e)->signalled.load());
    }
    EXPECT_EQ(t.cache->pending_count(), 0U);
    for (auto& m : part->missing) {
        EXPECT_EQ(t.cache->status_of_slot(m.slot), SlotStatus::READY);
    }
    EXPECT_EQ(sched.stats().demand_loads, 2U);
}

TEST(moe_lookahead_scheduler, event_is_attached_before_demand_returns) {
    auto loader = std::make_shared<FakeLoader>();
    loader->hold_completions();
    PrefetchScheduler sched(loader, 2);
    auto t = make_target(0, 4);
    auto part = t.cache->partition_3way({5});
    auto events = sched.submit_demand(t, part->missing);
    // a second partition of the same expert (in flight) must find exactly the event handed back to the caller
    auto again = t.cache->partition_3way({5});
    ASSERT_TRUE(again.has_value());
    ASSERT_EQ(again->inflight.size(), 1U);
    EXPECT_EQ(again->inflight[0].ready_event.get(), events[0].get());
    loader->open_gate();
    sched.quiesce();
}

TEST(moe_lookahead_scheduler, demand_goes_before_queued_speculative_work) {
    auto loader = std::make_shared<FakeLoader>();
    loader->hold_completions();
    PrefetchScheduler sched(loader, 1);  // one chunk: strictly one load in flight, the rest queues
    auto spec_target = make_target(5, 16);
    auto cur_target = make_target(4, 16);

    sched.submit_speculative(spec_target, {10, 11, 12, 13}, 8);
    ASSERT_TRUE(loader->wait_started(1));  // 10 is in flight and holds the only chunk

    auto part = cur_target.cache->partition_3way({99});
    auto ev = sched.submit_demand(cur_target, part->missing);
    loader->open_gate();
    sched.quiesce();

    auto order = loader->order();
    ASSERT_GE(order.size(), 2U);
    EXPECT_EQ(order[0].expert, 10U);
    EXPECT_EQ(order[1].expert, 99U) << "demand must overtake the speculative job after the load already in flight";
    EXPECT_TRUE(fake(ev[0])->signalled.load());
}

TEST(moe_lookahead_scheduler, speculative_job_of_a_started_layer_is_dropped) {
    auto loader = std::make_shared<FakeLoader>();
    loader->hold_completions();
    PrefetchScheduler sched(loader, 1);
    auto t5 = make_target(5, 16);
    auto t6 = make_target(6, 16);

    sched.submit_speculative(t5, {1, 2, 3}, 8);
    ASSERT_TRUE(loader->wait_started(1));
    sched.submit_speculative(t6, {20, 21}, 8);
    sched.set_cursor(6);  // layer 6 started computing: its own and layer 5's remaining predictions are stale
    loader->open_gate();
    sched.quiesce();

    for (auto& l : loader->order()) {
        EXPECT_NE(l.layer, 6U);
        EXPECT_FALSE(l.layer == 5U && l.expert != 1U) << "remaining layer-5 predictions must be dropped";
    }
    EXPECT_GE(sched.stats().speculative_dropped, 4U);
}

TEST(moe_lookahead_scheduler, speculative_respects_the_pending_quota) {
    auto loader = std::make_shared<FakeLoader>();
    loader->hold_completions();
    PrefetchScheduler sched(loader, 4);
    auto t = make_target(1, 16);
    sched.submit_speculative(t, {1, 2, 3, 4, 5, 6}, 2);
    ASSERT_TRUE(loader->wait_started(2));
    std::this_thread::sleep_for(std::chrono::milliseconds(80));
    EXPECT_EQ(loader->order().size(), 2U);  // quota of 2 in flight, the third reservation is refused
    EXPECT_EQ(t.cache->pending_count(), 2U);
    loader->open_gate();
    sched.wait_idle();
}

TEST(moe_lookahead_scheduler, staging_ring_bounds_inflight_loads_and_chunks_are_not_reused_early) {
    auto loader = std::make_shared<FakeLoader>();
    PrefetchScheduler sched(loader, 3);
    auto t = make_target(0, 32);
    std::vector<uint32_t> all;
    for (uint32_t i = 0; i < 24; ++i) {
        all.push_back(i);
    }
    auto part = t.cache->partition_3way(all);
    ASSERT_TRUE(part.has_value());
    sched.submit_demand(t, part->missing);
    sched.quiesce();
    EXPECT_LE(loader->max_chunks(), 3U);
    EXPECT_EQ(loader->order().size(), 24U);
    EXPECT_EQ(t.cache->pending_count(), 0U);
}

TEST(moe_lookahead_scheduler, load_failure_signals_event_invalidates_slot_and_reports_error) {
    auto loader = std::make_shared<FakeLoader>();
    loader->fail_on(7);
    PrefetchScheduler sched(loader, 2);
    auto t = make_target(0, 4);
    auto part = t.cache->partition_3way({7, 8});
    auto ev = sched.submit_demand(t, part->missing);
    sched.quiesce();

    for (auto& e : ev) {
        EXPECT_TRUE(fake(e)->signalled.load()) << "device work waiting on the event must never hang";
    }
    EXPECT_EQ(t.cache->pending_count(), 0U);
    EXPECT_THROW(sched.rethrow_if_error(), std::runtime_error);
    EXPECT_NO_THROW(sched.rethrow_if_error());  // reported once
    EXPECT_EQ(sched.stats().failed_loads, 1U);

    // The failed slot is pinned by the current layer, so the entry stays but its content is invalid: it is reloaded.
    std::vector<size_t> pins;
    for (auto& m : part->missing) {
        pins.push_back(m.slot);
    }
    t.cache->unpin(pins, nullptr);
    auto next = t.cache->partition_3way({7, 8});
    ASSERT_TRUE(next.has_value());
    EXPECT_EQ(next->ready.size(), 1U);    // 8 loaded fine
    EXPECT_EQ(next->missing.size(), 1U);  // 7 must be loaded again
    EXPECT_EQ(next->missing[0].expert, 7U);
}

TEST(moe_lookahead_scheduler, speculative_failure_frees_the_slot) {
    auto loader = std::make_shared<FakeLoader>();
    loader->fail_on(3);
    PrefetchScheduler sched(loader, 2);
    auto t = make_target(2, 4);
    sched.submit_speculative(t, {3, 4}, 4);
    sched.wait_idle();
    EXPECT_FALSE(t.cache->find_slot(3).has_value());
    EXPECT_TRUE(t.cache->find_slot(4).has_value());
    EXPECT_THROW(sched.rethrow_if_error(), std::runtime_error);
}

TEST(moe_lookahead_scheduler, cancel_drops_pending_speculative_jobs) {
    auto loader = std::make_shared<FakeLoader>();
    loader->hold_completions();
    PrefetchScheduler sched(loader, 1);
    auto t = make_target(3, 16);
    sched.submit_speculative(t, {1, 2, 3, 4}, 8);
    ASSERT_TRUE(loader->wait_started(1));
    sched.cancel_speculative();
    loader->open_gate();
    sched.quiesce();
    EXPECT_EQ(loader->order().size(), 1U);
}

TEST(moe_lookahead_scheduler, destructor_drains_demand_and_never_leaves_an_event_unsignalled) {
    auto loader = std::make_shared<FakeLoader>();
    loader->hold_completions();
    std::vector<EventHandle> events;
    auto t = make_target(0, 16);
    std::thread opener;
    {
        PrefetchScheduler sched(loader, 2);
        auto part = t.cache->partition_3way({1, 2, 3, 4, 5});
        events = sched.submit_demand(t, part->missing);
        sched.submit_speculative(make_target(1, 16), {9, 10, 11}, 8);
        ASSERT_TRUE(loader->wait_started(1));
        opener = std::thread([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(60));
            loader->open_gate();
        });
        // the destructor runs here while copies are still held in flight and must wait for them
    }
    opener.join();
    for (auto& e : events) {
        EXPECT_TRUE(fake(e)->signalled.load());
    }
}

TEST(moe_lookahead_scheduler, stress_mixed_demand_and_speculative_keeps_cache_consistent) {
    auto loader = std::make_shared<FakeLoader>();
    PrefetchScheduler sched(loader, 3);
    constexpr size_t layers = 4;
    constexpr size_t cap = 10;
    std::vector<PrefetchTarget> targets;
    for (size_t l = 0; l < layers; ++l) {
        targets.push_back(make_target(l, cap));
    }
    std::mt19937 rng(99);
    for (int iter = 0; iter < 400; ++iter) {
        const size_t l = iter % layers;
        sched.set_cursor(l);
        if (l + 1 < layers) {
            std::vector<uint32_t> pred;
            for (int i = 0; i < 4; ++i) {
                pred.push_back(rng() % 30);
            }
            sched.submit_speculative(targets[l + 1], pred, cap / 4);
        }
        std::vector<uint32_t> req;
        for (int i = 0; i < 4; ++i) {
            req.push_back(rng() % 30);
        }
        auto part = targets[l].cache->partition_3way(req);
        ASSERT_TRUE(part.has_value());
        auto ev = sched.submit_demand(targets[l], part->missing);
        std::vector<size_t> slots;
        for (auto* g : {&part->ready, &part->inflight, &part->missing}) {
            for (auto& s : *g) {
                slots.push_back(s.slot);
            }
        }
        for (auto& f : part->inflight) {
            ASSERT_TRUE(wait_for([&] {
                return fake(f.ready_event)->signalled.load();
            }));
        }
        for (auto& e : ev) {
            ASSERT_TRUE(wait_for([&] {
                return fake(e)->signalled.load();
            }));
        }
        targets[l].cache->unpin(slots, nullptr);
    }
    sched.quiesce();
    for (auto& t : targets) {
        EXPECT_EQ(t.cache->pending_count(), 0U);
        EXPECT_EQ(t.cache->pinned_count(), 0U);
        EXPECT_LE(t.cache->size(), cap);
    }
    EXPECT_NO_THROW(sched.rethrow_if_error());
}

namespace {
class ThrowingEventLoader : public FakeLoader {
public:
    EventHandle make_ready_event() override {
        if (m_throw.load()) {
            throw std::runtime_error("cannot create event");
        }
        return FakeLoader::make_ready_event();
    }
    std::atomic<bool> m_throw{false};
};
}  // namespace

TEST(moe_lookahead_scheduler, speculative_event_creation_failure_does_not_strand_a_reserved_slot) {
    auto loader = std::make_shared<ThrowingEventLoader>();
    PrefetchScheduler sched(loader, 2);
    auto t = make_target(2, 4);
    loader->m_throw = true;
    sched.submit_speculative(t, {3}, 4);
    sched.wait_idle();
    EXPECT_EQ(t.cache->pending_count(), 0U);  // no slot left PENDING without an event
    EXPECT_FALSE(t.cache->find_slot(3).has_value());
    EXPECT_THROW(sched.rethrow_if_error(), std::runtime_error);
    // the main thread must not block on that expert
    auto p = std::async(std::launch::async, [&] { return t.cache->partition_3way({3}); });
    ASSERT_EQ(p.wait_for(std::chrono::seconds(2)), std::future_status::ready);
    EXPECT_EQ(p.get()->missing.size(), 1U);
}
