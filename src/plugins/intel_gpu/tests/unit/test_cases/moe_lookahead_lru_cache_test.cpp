// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <atomic>
#include <chrono>
#include <future>
#include <random>
#include <set>
#include <thread>
#include <vector>

#include "ocl_v2/moe/lru_cache.hpp"
#include "test_utils/test_utils.h"

using ov::intel_gpu::ocl::moe::EventHandle;
using ov::intel_gpu::ocl::moe::ExpertPartition;
using ov::intel_gpu::ocl::moe::ExpertSlot;
using ov::intel_gpu::ocl::moe::LRUCache;
using ov::intel_gpu::ocl::moe::SlotStatus;

namespace {

// LRUCache never dereferences an event, it only stores and returns it, so a tagged non-null handle is enough here.
EventHandle make_event(int tag) {
    static std::vector<std::shared_ptr<int>> owners;
    static std::mutex m;
    std::lock_guard<std::mutex> lock(m);
    owners.push_back(std::make_shared<int>(tag));
    return EventHandle(owners.back(), reinterpret_cast<cldnn::event*>(owners.back().get()));
}

std::vector<size_t> slots_of(const ExpertPartition& p) {
    std::vector<size_t> s;
    for (const auto* g : {&p.ready, &p.inflight, &p.missing}) {
        for (const auto& e : *g) {
            s.push_back(e.slot);
        }
    }
    return s;
}

std::vector<uint32_t> experts_of(const ExpertPartition& p) {
    std::vector<uint32_t> s;
    for (const auto* g : {&p.ready, &p.inflight, &p.missing}) {
        for (const auto& e : *g) {
            s.push_back(e.expert);
        }
    }
    return s;
}

// Completes the load of every M of a partition the way the prefetcher does.
void complete_missing(LRUCache& cache, const ExpertPartition& p) {
    for (const auto& m : p.missing) {
        cache.set_ready_event(m.slot, make_event(1));
        cache.mark_ready(m.slot);
    }
}

}  // namespace

TEST(moe_lookahead_lru, all_missing_on_cold_cache) {
    LRUCache cache(4);
    auto p = cache.partition_3way({3, 1, 2});
    ASSERT_TRUE(p.has_value());
    EXPECT_TRUE(p->ready.empty());
    EXPECT_TRUE(p->inflight.empty());
    ASSERT_EQ(p->missing.size(), 3U);
    std::set<size_t> slots;
    for (const auto& m : p->missing) {
        slots.insert(m.slot);
        EXPECT_TRUE(cache.is_pinned(m.slot));
        EXPECT_EQ(cache.status_of_slot(m.slot), SlotStatus::PENDING_LOAD);
        EXPECT_EQ(cache.find_slot(m.expert).value(), m.slot);
    }
    EXPECT_EQ(slots.size(), 3U);
    EXPECT_EQ(cache.pending_count(), 3U);
    EXPECT_EQ(cache.pinned_count(), 3U);
}

TEST(moe_lookahead_lru, duplicates_are_collapsed) {
    LRUCache cache(4);
    auto p = cache.partition_3way({5, 5, 7, 5, 7});
    ASSERT_TRUE(p.has_value());
    EXPECT_EQ(p->missing.size(), 2U);
    EXPECT_EQ(cache.size(), 2U);
}

TEST(moe_lookahead_lru, loaded_expert_is_ready_afterwards) {
    LRUCache cache(4);
    auto p1 = cache.partition_3way({0, 1});
    ASSERT_TRUE(p1.has_value());
    complete_missing(cache, *p1);
    std::vector<size_t> pins;
    for (const auto& m : p1->missing) {
        pins.push_back(m.slot);
    }
    cache.unpin(pins, make_event(2));
    EXPECT_EQ(cache.pinned_count(), 0U);
    EXPECT_EQ(cache.pending_count(), 0U);

    auto p2 = cache.partition_3way({1, 2, 0});
    ASSERT_TRUE(p2.has_value());
    EXPECT_EQ(p2->ready.size(), 2U);
    EXPECT_TRUE(p2->inflight.empty());
    ASSERT_EQ(p2->missing.size(), 1U);
    EXPECT_EQ(p2->missing[0].expert, 2U);
}

TEST(moe_lookahead_lru, sets_are_disjoint_and_cover_the_request) {
    LRUCache cache(6);
    // 0,1 resident+valid, 2 in flight (speculative), 3,4 missing
    auto warm = cache.partition_3way({0, 1});
    ASSERT_TRUE(warm.has_value());
    complete_missing(cache, *warm);
    cache.unpin({warm->missing[0].slot, warm->missing[1].slot}, nullptr);
    auto spec = cache.reserve_speculative(2, 4);
    ASSERT_TRUE(spec.has_value());
    cache.set_ready_event(spec->slot, make_event(7));

    auto p = cache.partition_3way({0, 1, 2, 3, 4});
    ASSERT_TRUE(p.has_value());
    EXPECT_EQ(p->ready.size(), 2U);
    ASSERT_EQ(p->inflight.size(), 1U);
    EXPECT_EQ(p->inflight[0].expert, 2U);
    EXPECT_EQ(p->missing.size(), 2U);

    auto e = experts_of(*p);
    std::sort(e.begin(), e.end());
    EXPECT_EQ(e, (std::vector<uint32_t>{0, 1, 2, 3, 4}));
    auto s = slots_of(*p);
    std::set<size_t> su(s.begin(), s.end());
    EXPECT_EQ(su.size(), s.size());
}

TEST(moe_lookahead_lru, inflight_reuses_the_existing_event_and_never_reloads) {
    LRUCache cache(4);
    auto spec = cache.reserve_speculative(9, 4);
    ASSERT_TRUE(spec.has_value());
    auto ev = make_event(11);
    cache.set_ready_event(spec->slot, ev);

    auto p = cache.partition_3way({9});
    ASSERT_TRUE(p.has_value());
    EXPECT_TRUE(p->missing.empty());
    ASSERT_EQ(p->inflight.size(), 1U);
    EXPECT_EQ(p->inflight[0].slot, spec->slot);
    EXPECT_EQ(p->inflight[0].ready_event.get(), ev.get());
    EXPECT_EQ(cache.pending_count(), 1U);  // still the one load issued by the speculative path (I3)
    EXPECT_TRUE(cache.is_pinned(spec->slot));
}

TEST(moe_lookahead_lru, capacity_failure_changes_nothing) {
    LRUCache cache(3);
    EXPECT_FALSE(cache.partition_3way({0, 1, 2, 3}).has_value());
    EXPECT_EQ(cache.size(), 0U);
    EXPECT_EQ(cache.pinned_count(), 0U);
    EXPECT_EQ(cache.pending_count(), 0U);

    // all slots pinned by the current layer: a request needing a new slot must fail and must not leak a pin
    auto p = cache.partition_3way({0, 1, 2});
    ASSERT_TRUE(p.has_value());
    complete_missing(cache, *p);
    EXPECT_FALSE(cache.partition_3way({0, 3}).has_value());
    EXPECT_EQ(cache.pinned_count(), 3U);  // unchanged: the failed call released the pin it took on expert 0
    EXPECT_EQ(cache.size(), 3U);
}

TEST(moe_lookahead_lru, pinned_slots_are_never_evicted) {
    LRUCache cache(2);
    auto p = cache.partition_3way({0, 1});
    ASSERT_TRUE(p.has_value());
    complete_missing(cache, *p);  // READY but still pinned
    EXPECT_FALSE(cache.reserve_speculative(2, 8).has_value());
    EXPECT_FALSE(cache.partition_3way({2}).has_value());
    EXPECT_TRUE(cache.find_slot(0).has_value());
    EXPECT_TRUE(cache.find_slot(1).has_value());

    cache.unpin({p->missing[0].slot, p->missing[1].slot}, nullptr);
    EXPECT_TRUE(cache.reserve_speculative(2, 8).has_value());
}

TEST(moe_lookahead_lru, pending_slots_are_never_evicted_by_demand) {
    LRUCache cache(2);
    auto a = cache.reserve_speculative(0, 8);
    auto b = cache.reserve_speculative(1, 8);
    ASSERT_TRUE(a.has_value() && b.has_value());
    cache.set_ready_event(a->slot, make_event(1));
    cache.set_ready_event(b->slot, make_event(2));
    // both slots hold loads in flight and nothing is evictable
    EXPECT_FALSE(cache.partition_3way({5}).has_value());
    EXPECT_TRUE(cache.find_slot(0).has_value());
    EXPECT_TRUE(cache.find_slot(1).has_value());
}

TEST(moe_lookahead_lru, speculative_quota_and_duplicates) {
    LRUCache cache(8);
    EXPECT_TRUE(cache.reserve_speculative(0, 2).has_value());
    EXPECT_TRUE(cache.reserve_speculative(1, 2).has_value());
    EXPECT_FALSE(cache.reserve_speculative(2, 2).has_value());  // quota reached
    EXPECT_FALSE(cache.reserve_speculative(1, 2).has_value());  // already in flight
    cache.mark_ready(cache.find_slot(0).value());
    EXPECT_FALSE(cache.reserve_speculative(0, 8).has_value());  // already resident
    EXPECT_TRUE(cache.reserve_speculative(2, 2).has_value());   // one slot of quota is free again
}

TEST(moe_lookahead_lru, speculative_entries_are_first_eviction_candidates) {
    LRUCache cache(2);
    auto warm = cache.partition_3way({0});
    ASSERT_TRUE(warm.has_value());
    complete_missing(cache, *warm);
    cache.unpin({warm->missing[0].slot}, nullptr);

    auto spec = cache.reserve_speculative(1, 8);  // wrong prediction
    ASSERT_TRUE(spec.has_value());
    cache.mark_ready(spec->slot);

    auto p = cache.partition_3way({2});  // needs a slot: the mispredicted entry goes, not the proven one
    ASSERT_TRUE(p.has_value());
    EXPECT_TRUE(cache.find_slot(0).has_value());
    EXPECT_FALSE(cache.find_slot(1).has_value());
}

TEST(moe_lookahead_lru, war_event_follows_the_slot) {
    LRUCache cache(1);
    auto p1 = cache.partition_3way({0});
    ASSERT_TRUE(p1.has_value());
    EXPECT_EQ(p1->missing[0].war_event, nullptr);  // never used, nothing to wait for
    complete_missing(cache, *p1);
    auto use = make_event(42);
    cache.unpin({p1->missing[0].slot}, use);

    auto p2 = cache.partition_3way({1});  // overwrites the only slot
    ASSERT_TRUE(p2.has_value());
    ASSERT_EQ(p2->missing.size(), 1U);
    EXPECT_EQ(p2->missing[0].slot, p1->missing[0].slot);
    EXPECT_EQ(p2->missing[0].war_event.get(), use.get());  // I2: the load must wait for the last reader
}

TEST(moe_lookahead_lru, abort_pending_frees_the_slot) {
    LRUCache cache(1);
    auto spec = cache.reserve_speculative(3, 4);
    ASSERT_TRUE(spec.has_value());
    EXPECT_EQ(cache.pending_count(), 1U);
    cache.abort_pending(spec->slot);
    EXPECT_EQ(cache.pending_count(), 0U);
    EXPECT_FALSE(cache.find_slot(3).has_value());
    EXPECT_EQ(cache.size(), 0U);
    EXPECT_TRUE(cache.reserve_speculative(4, 4).has_value());
}

TEST(moe_lookahead_lru, stale_resident_entry_is_reloaded_in_place) {
    LRUCache cache(2);
    auto [slot, hit] = cache.get_lru_item(5);  // legacy insert that was never filled
    ASSERT_FALSE(hit);
    auto p = cache.partition_3way({5});
    ASSERT_TRUE(p.has_value());
    EXPECT_TRUE(p->ready.empty());
    ASSERT_EQ(p->missing.size(), 1U);
    EXPECT_EQ(p->missing[0].slot, slot);
    EXPECT_EQ(cache.pending_count(), 1U);
}

TEST(moe_lookahead_lru, legacy_path_is_unchanged_without_lookahead) {
    LRUCache cache(2);
    auto [s0, h0] = cache.get_lru_item(0);
    cache.set_filled(s0);
    auto [s1, h1] = cache.get_lru_item(1);
    cache.set_filled(s1);
    (void)h0;
    (void)h1;
    auto [s2, h2] = cache.get_lru_item(2);  // evicts 0
    EXPECT_FALSE(h2);
    EXPECT_EQ(s2, s0);
    EXPECT_FALSE(cache.find_slot(0).has_value());
}

TEST(moe_lookahead_lru, wait_no_pending_blocks_until_the_load_completes) {
    LRUCache cache(2);
    auto spec = cache.reserve_speculative(0, 4);
    ASSERT_TRUE(spec.has_value());
    std::atomic<bool> done{false};
    std::thread waiter([&] {
        cache.wait_no_pending();
        done = true;
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_FALSE(done.load());
    cache.mark_ready(spec->slot);
    waiter.join();
    EXPECT_TRUE(done.load());
}

TEST(moe_lookahead_lru, partition_waits_for_the_event_of_a_load_that_is_still_being_issued) {
    LRUCache cache(2);
    auto spec = cache.reserve_speculative(0, 4);  // reserved, worker has not issued the copy yet: no event
    ASSERT_TRUE(spec.has_value());
    auto ev = make_event(5);
    auto fut = std::async(std::launch::async, [&] {
        return cache.partition_3way({0});
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_EQ(fut.wait_for(std::chrono::milliseconds(0)), std::future_status::timeout);
    cache.set_ready_event(spec->slot, ev);
    auto p = fut.get();
    ASSERT_TRUE(p.has_value());
    ASSERT_EQ(p->inflight.size(), 1U);
    EXPECT_EQ(p->inflight[0].ready_event.get(), ev.get());
}

TEST(moe_lookahead_lru, randomized_invariants_with_background_loader) {
    constexpr size_t cap = 12;
    constexpr uint32_t num_experts = 40;
    LRUCache cache(cap);

    std::atomic<bool> stop{false};
    std::mutex q_mutex;
    std::vector<size_t> to_complete;  // slots whose "DMA" has been issued
    // Background loader: speculative prefetch of random experts, completing them asynchronously.
    std::thread loader([&] {
        std::mt19937 rng(7);
        while (!stop.load()) {
            const uint32_t e = rng() % num_experts;
            if (auto r = cache.reserve_speculative(e, 3)) {
                cache.set_ready_event(r->slot, make_event(static_cast<int>(e)));
                std::lock_guard<std::mutex> lock(q_mutex);
                to_complete.push_back(r->slot);
            }
            std::vector<size_t> done;
            {
                std::lock_guard<std::mutex> lock(q_mutex);
                done.swap(to_complete);
            }
            for (auto s : done) {
                cache.mark_ready(s);
            }
            std::this_thread::yield();
        }
    });

    std::mt19937 rng(1234);
    for (int iter = 0; iter < 3000; ++iter) {
        std::vector<uint32_t> req;
        const size_t n = 1 + rng() % 8;
        for (size_t i = 0; i < n; ++i) {
            req.push_back(rng() % num_experts);
        }
        auto p = cache.partition_3way(req);
        ASSERT_TRUE(p.has_value());  // 8 unique <= cap and the loader never pins

        std::set<uint32_t> want(req.begin(), req.end());
        auto e = experts_of(*p);
        std::set<uint32_t> got(e.begin(), e.end());
        ASSERT_EQ(e.size(), got.size()) << "an expert is in two sets";
        ASSERT_EQ(want, got);
        auto s = slots_of(*p);
        ASSERT_EQ(std::set<size_t>(s.begin(), s.end()).size(), s.size()) << "two experts share a slot";
        for (const auto& m : p->missing) {
            cache.set_ready_event(m.slot, make_event(1));
            cache.mark_ready(m.slot);
        }
        for (const auto& f : p->inflight) {
            ASSERT_NE(f.ready_event, nullptr);
        }
        cache.unpin(s, make_event(iter));
        ASSERT_EQ(cache.pinned_count(), 0U);
    }
    stop = true;
    loader.join();
    {
        std::lock_guard<std::mutex> lock(q_mutex);
        for (auto s : to_complete) {
            cache.mark_ready(s);
        }
    }
    cache.wait_no_pending();
    EXPECT_EQ(cache.pending_count(), 0U);
    EXPECT_LE(cache.size(), cap);
}

TEST(moe_lookahead_lru, disabled_speculation_refuses_and_wait_closes_the_legacy_race) {
    LRUCache cache(4);
    auto spec = cache.reserve_speculative(0, 4);
    ASSERT_TRUE(spec.has_value());
    cache.set_speculation_enabled(false);
    EXPECT_FALSE(cache.reserve_speculative(1, 4).has_value());
    cache.mark_ready(spec->slot);
    cache.wait_no_pending();  // returns immediately, nothing in flight and nothing can start
    EXPECT_EQ(cache.pending_count(), 0U);
    auto [slot, hit] = cache.get_lru_item(0);  // legacy lookup sees the filled slot as a hit
    EXPECT_TRUE(hit);
    EXPECT_EQ(slot, spec->slot);
    cache.set_speculation_enabled(true);
    EXPECT_TRUE(cache.reserve_speculative(1, 4).has_value());
}
