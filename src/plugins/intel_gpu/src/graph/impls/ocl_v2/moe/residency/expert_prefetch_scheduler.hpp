// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "../lru_cache.hpp"

namespace ov::intel_gpu::ocl::moe::lookahead {

/// A layer's expert cache plus whatever the loader needs to find the layer's device buffers (opaque to the scheduler).
struct PrefetchTarget {
    std::shared_ptr<LRUCache> cache;
    std::shared_ptr<void> owner;
    size_t layer_index = 0;
};

/// Backend that moves one expert's weights from storage into a device slot. The scheduler owns *when* and *in which
/// order* things are loaded, the loader owns *how*. This split keeps the scheduling policy testable without a GPU.
class IExpertLoader {
public:
    using Completion = std::function<void()>;
    virtual ~IExpertLoader() = default;

    /// Creates an event that is signalled once the load of a slot has finished. Callable from any thread.
    virtual EventHandle make_ready_event() = 0;

    /// Signals an event created by make_ready_event(). Must not throw. Called on success and on failure, so that device
    /// work that already depends on the event can never deadlock.
    virtual void signal(const EventHandle& event) noexcept = 0;

    /// Reads the expert from storage into staging chunk `staging_chunk` and enqueues the asynchronous copy into `slot`.
    /// The copy must not start before `war_event` (the last kernel that read the previous content of the slot).
    /// Returns a completion that blocks until the copy has finished. May throw.
    virtual Completion begin_load(const PrefetchTarget& target,
                                  uint32_t expert,
                                  size_t slot,
                                  const EventHandle& war_event,
                                  size_t staging_chunk) = 0;
};

/// Two-priority asynchronous expert loader:
///  * demand loads (experts the current layer needs now) always go before speculative loads;
///  * a speculative job is processed expert by expert and yields to demand between two experts, so the head-of-line
///    blocking a demand load sees is bounded by the number of staging chunks in flight;
///  * speculative jobs of a layer that has already started computing are dropped (prefetch cursor);
///  * the number of loads in flight is bounded by the staging ring, and a chunk is only reused after its copy finished.
class PrefetchScheduler {
public:
    explicit PrefetchScheduler(std::shared_ptr<IExpertLoader> loader, size_t staging_chunks = 2);
    ~PrefetchScheduler();

    PrefetchScheduler(const PrefetchScheduler&) = delete;
    PrefetchScheduler& operator=(const PrefetchScheduler&) = delete;

    /// Queues the loads of the M experts returned by LRUCache::partition_3way. Attaches the ready event of each slot
    /// before returning, and returns those events (parallel to `missing`). Never blocks on I/O.
    std::vector<EventHandle> submit_demand(const PrefetchTarget& target, const std::vector<ExpertSlot>& missing);

    /// Queues a speculative load of `predicted` (in priority order) into `target`. At most `max_pending` loads of the
    /// target cache are in flight at once, the rest of the prediction is discarded. Never blocks on I/O.
    void submit_speculative(const PrefetchTarget& target, std::vector<uint32_t> predicted, size_t max_pending);

    /// Layer `layer` has started to compute: speculative work for it or for an earlier layer is useless.
    void set_cursor(size_t layer);

    /// Drops every speculative job that has not started yet (e.g. the request was cancelled).
    void cancel_speculative();

    /// Waits until nothing is queued, being issued or in flight. Speculative work already queued still runs.
    void wait_idle();

    /// Cancels speculative work and waits until nothing is queued or in flight.
    void quiesce();

    /// Rethrows (once) the first failure of a background load.
    void rethrow_if_error();

    struct Stats {
        uint64_t demand_loads = 0;
        uint64_t speculative_loads = 0;
        uint64_t speculative_dropped = 0;
        uint64_t failed_loads = 0;
        size_t max_inflight = 0;
    };
    Stats stats() const;

private:
    struct DemandItem {
        PrefetchTarget target;
        uint32_t expert = 0;
        size_t slot = 0;
        EventHandle war_event;
        EventHandle ready_event;
    };
    struct SpeculativeJob {
        PrefetchTarget target;
        std::vector<uint32_t> predicted;
        size_t max_pending = 0;
        uint64_t generation = 0;
    };
    struct CompletionRecord {
        std::shared_ptr<LRUCache> cache;
        size_t slot = 0;
        EventHandle ready_event;
        size_t chunk = 0;
        IExpertLoader::Completion completion;
    };

    void worker_loop();
    void completion_loop();
    void run_speculative(const SpeculativeJob& job);
    void run_load(size_t chunk, const PrefetchTarget& target, uint32_t expert, size_t slot, const EventHandle& war, const EventHandle& ev);
    bool pop_demand(DemandItem& out);
    bool acquire_chunk(size_t& chunk);
    void release_chunk(size_t chunk);
    void record_error(std::exception_ptr e);
    void fail_load(const std::shared_ptr<LRUCache>& cache, size_t slot, const EventHandle& ev);
    bool is_idle_locked() const;

    std::shared_ptr<IExpertLoader> m_loader;

    mutable std::mutex m_mutex;
    std::condition_variable m_work_cv;
    std::condition_variable m_idle_cv;
    std::deque<DemandItem> m_demand;
    std::deque<SpeculativeJob> m_speculative;
    bool m_stop = false;
    size_t m_active_jobs = 0;  // jobs popped by the worker and not finished yet
    uint64_t m_generation = 0;
    std::atomic<size_t> m_cursor{0};
    std::atomic<bool> m_cursor_valid{false};

    std::mutex m_chunk_mutex;
    std::condition_variable m_chunk_cv;
    std::vector<size_t> m_free_chunks;
    size_t m_total_chunks = 0;

    std::mutex m_completion_mutex;
    std::condition_variable m_completion_cv;
    std::deque<CompletionRecord> m_completions;
    bool m_completion_stop = false;

    mutable std::mutex m_error_mutex;
    std::exception_ptr m_error;

    mutable std::mutex m_stats_mutex;
    Stats m_stats;

    std::thread m_worker;
    std::thread m_completer;
};

}  // namespace ov::intel_gpu::ocl::moe::lookahead
