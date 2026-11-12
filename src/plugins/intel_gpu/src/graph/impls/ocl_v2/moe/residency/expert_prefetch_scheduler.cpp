// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "expert_prefetch_scheduler.hpp"

#include <algorithm>
#include <utility>

namespace ov::intel_gpu::ocl::moe::lookahead {

PrefetchScheduler::PrefetchScheduler(std::shared_ptr<IExpertLoader> loader, size_t staging_chunks)
    : m_loader(std::move(loader)),
      m_total_chunks(std::max<size_t>(1, staging_chunks)) {
    m_free_chunks.reserve(m_total_chunks);
    for (size_t i = m_total_chunks; i > 0; --i) {
        m_free_chunks.push_back(i - 1);
    }
    m_completer = std::thread([this] {
        completion_loop();
    });
    m_worker = std::thread([this] {
        worker_loop();
    });
}

PrefetchScheduler::~PrefetchScheduler() {
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_stop = true;
        for (const auto& j : m_speculative) {
            m_stats.speculative_dropped += j.predicted.size();
        }
        m_speculative.clear();
        ++m_generation;
    }
    m_work_cv.notify_all();
    // The worker still serves the demand queue before it exits, so every ready event handed out is signalled.
    if (m_worker.joinable()) {
        m_worker.join();
    }
    {
        std::lock_guard<std::mutex> lock(m_completion_mutex);
        m_completion_stop = true;
    }
    m_completion_cv.notify_all();
    if (m_completer.joinable()) {
        m_completer.join();
    }
}

std::vector<EventHandle> PrefetchScheduler::submit_demand(const PrefetchTarget& target, const std::vector<ExpertSlot>& missing) {
    std::vector<EventHandle> events;
    events.reserve(missing.size());
    std::vector<DemandItem> items;
    items.reserve(missing.size());
    try {
        for (const auto& m : missing) {
            auto ev = m_loader->make_ready_event();
            // The event is attached before the item is queued, so a concurrent partition_3way that sees the slot as in
            // flight always finds an event to wait for.
            target.cache->set_ready_event(m.slot, ev);
            events.push_back(ev);
            items.push_back(DemandItem{target, m.expert, m.slot, m.war_event, ev});
        }
    } catch (...) {
        // Nothing was queued: release the slots reserved by partition_3way so that nobody waits for a load that will
        // never happen, and never leave an attached event unsignalled.
        for (const auto& m : missing) {
            target.cache->abort_pending(m.slot);
        }
        for (const auto& ev : events) {
            m_loader->signal(ev);
        }
        throw;
    }
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        for (auto& it : items) {
            m_demand.push_back(std::move(it));
        }
    }
    m_work_cv.notify_one();
    return events;
}

void PrefetchScheduler::submit_speculative(const PrefetchTarget& target, std::vector<uint32_t> predicted, size_t max_pending) {
    if (predicted.empty()) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_stop) {
            return;
        }
        m_speculative.push_back(SpeculativeJob{target, std::move(predicted), max_pending, m_generation});
    }
    m_work_cv.notify_one();
}

void PrefetchScheduler::set_cursor(size_t layer) {
    m_cursor.store(layer, std::memory_order_release);
    m_cursor_valid.store(true, std::memory_order_release);
}

void PrefetchScheduler::cancel_speculative() {
    std::lock_guard<std::mutex> lock(m_mutex);
    for (const auto& j : m_speculative) {
        m_stats.speculative_dropped += j.predicted.size();
    }
    m_speculative.clear();
    ++m_generation;
}

bool PrefetchScheduler::is_idle_locked() const {
    return m_demand.empty() && m_speculative.empty() && m_active_jobs == 0;
}

void PrefetchScheduler::wait_idle() {
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_idle_cv.wait(lock, [this] {
            return is_idle_locked();
        });
    }
    // Completions finish after the worker is idle: wait until every chunk is back.
    std::unique_lock<std::mutex> lock(m_chunk_mutex);
    m_chunk_cv.wait(lock, [this] {
        return m_free_chunks.size() == m_total_chunks;
    });
}

void PrefetchScheduler::quiesce() {
    cancel_speculative();
    wait_idle();
}

void PrefetchScheduler::rethrow_if_error() {
    std::exception_ptr e;
    {
        std::lock_guard<std::mutex> lock(m_error_mutex);
        e = std::exchange(m_error, nullptr);
    }
    if (e) {
        std::rethrow_exception(e);
    }
}

PrefetchScheduler::Stats PrefetchScheduler::stats() const {
    std::lock_guard<std::mutex> lock(m_stats_mutex);
    return m_stats;
}

void PrefetchScheduler::record_error(std::exception_ptr e) {
    std::lock_guard<std::mutex> lock(m_error_mutex);
    if (!m_error) {
        m_error = std::move(e);
    }
}

bool PrefetchScheduler::pop_demand(DemandItem& out) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_demand.empty()) {
        return false;
    }
    out = std::move(m_demand.front());
    m_demand.pop_front();
    return true;
}

bool PrefetchScheduler::acquire_chunk(size_t& chunk) {
    std::unique_lock<std::mutex> lock(m_chunk_mutex);
    // Completions are processed by their own thread, so this wait is bounded by the copies already in flight.
    m_chunk_cv.wait(lock, [this] {
        return !m_free_chunks.empty();
    });
    chunk = m_free_chunks.back();
    m_free_chunks.pop_back();
    const size_t inflight = m_total_chunks - m_free_chunks.size();
    {
        std::lock_guard<std::mutex> stats_lock(m_stats_mutex);
        m_stats.max_inflight = std::max(m_stats.max_inflight, inflight);
    }
    return true;
}

void PrefetchScheduler::release_chunk(size_t chunk) {
    {
        std::lock_guard<std::mutex> lock(m_chunk_mutex);
        m_free_chunks.push_back(chunk);
    }
    m_chunk_cv.notify_all();
}

void PrefetchScheduler::fail_load(const std::shared_ptr<LRUCache>& cache, size_t slot, const EventHandle& ev) {
    cache->abort_pending(slot);
    // Device work that already depends on the event must not hang. It may read an invalid slot, which is why the error
    // is reported to the main thread (rethrow_if_error) and the slot is invalidated above.
    m_loader->signal(ev);
    std::lock_guard<std::mutex> lock(m_stats_mutex);
    ++m_stats.failed_loads;
}

void PrefetchScheduler::run_load(size_t chunk,
                                 const PrefetchTarget& target,
                                 uint32_t expert,
                                 size_t slot,
                                 const EventHandle& war,
                                 const EventHandle& ev) {
    try {
        auto completion = m_loader->begin_load(target, expert, slot, war, chunk);
        {
            std::lock_guard<std::mutex> lock(m_completion_mutex);
            m_completions.push_back(CompletionRecord{target.cache, slot, ev, chunk, std::move(completion)});
        }
        m_completion_cv.notify_one();
    } catch (...) {
        release_chunk(chunk);
        fail_load(target.cache, slot, ev);
        record_error(std::current_exception());
    }
}

void PrefetchScheduler::completion_loop() {
    for (;;) {
        CompletionRecord rec;
        {
            std::unique_lock<std::mutex> lock(m_completion_mutex);
            m_completion_cv.wait(lock, [this] {
                return m_completion_stop || !m_completions.empty();
            });
            if (m_completions.empty()) {
                return;  // stop requested and drained
            }
            rec = std::move(m_completions.front());
            m_completions.pop_front();
        }
        try {
            rec.completion();
            // Order matters: the event is signalled before the host state flips to READY, so a reader that sees READY
            // may rely on the data, and a reader that waits on the event is released only after the copy has finished.
            m_loader->signal(rec.ready_event);
            rec.cache->mark_ready(rec.slot);
        } catch (...) {
            fail_load(rec.cache, rec.slot, rec.ready_event);
            record_error(std::current_exception());
        }
        release_chunk(rec.chunk);
    }
}

void PrefetchScheduler::run_speculative(const SpeculativeJob& job) {
    for (size_t i = 0; i < job.predicted.size(); ++i) {
        // Take the staging chunk BEFORE reserving a slot or deciding what to load. Blocking on the ring is where time
        // passes, so everything that can change while we wait (new demand, the cursor moving, a cancel) is evaluated
        // only after the wait, never before it.
        size_t chunk = 0;
        acquire_chunk(chunk);

        // Demand always goes first: a speculative load never delays a load the current layer is waiting for by more
        // than the copies already in flight.
        DemandItem d;
        while (pop_demand(d)) {
            run_load(chunk, d.target, d.expert, d.slot, d.war_event, d.ready_event);
            {
                std::lock_guard<std::mutex> lock(m_stats_mutex);
                ++m_stats.demand_loads;
            }
            acquire_chunk(chunk);
        }

        bool stale = false;
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            stale = (job.generation != m_generation);
        }
        if (!stale && m_cursor_valid.load(std::memory_order_acquire) && job.target.layer_index <= m_cursor.load(std::memory_order_acquire)) {
            stale = true;  // that layer already started to compute, the prediction is of no use any more
        }
        if (stale) {
            release_chunk(chunk);
            std::lock_guard<std::mutex> lock(m_stats_mutex);
            m_stats.speculative_dropped += job.predicted.size() - i;
            return;
        }
        auto reservation = job.target.cache->reserve_speculative(job.predicted[i], job.max_pending);
        if (!reservation) {
            release_chunk(chunk);
            continue;  // already resident / in flight, quota reached, or nothing evictable
        }
        EventHandle ev;
        try {
            ev = m_loader->make_ready_event();
        } catch (...) {
            // The slot is reserved (PENDING without an event): free it, or partition_3way would wait for it forever.
            job.target.cache->abort_pending(reservation->slot);
            release_chunk(chunk);
            record_error(std::current_exception());
            return;
        }
        job.target.cache->set_ready_event(reservation->slot, ev);
        run_load(chunk, job.target, reservation->expert, reservation->slot, reservation->war_event, ev);
        std::lock_guard<std::mutex> lock(m_stats_mutex);
        ++m_stats.speculative_loads;
    }
}

void PrefetchScheduler::worker_loop() {
    for (;;) {
        DemandItem demand;
        SpeculativeJob job;
        bool is_demand = false;
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_work_cv.wait(lock, [this] {
                return m_stop || !m_demand.empty() || !m_speculative.empty();
            });
            if (!m_demand.empty()) {
                demand = std::move(m_demand.front());
                m_demand.pop_front();
                is_demand = true;
            } else if (!m_speculative.empty()) {
                job = std::move(m_speculative.front());
                m_speculative.pop_front();
            } else {
                return;  // stop requested, nothing left to serve
            }
            ++m_active_jobs;
        }
        if (is_demand) {
            size_t chunk = 0;
            acquire_chunk(chunk);
            run_load(chunk, demand.target, demand.expert, demand.slot, demand.war_event, demand.ready_event);
            std::lock_guard<std::mutex> lock(m_stats_mutex);
            ++m_stats.demand_loads;
        } else {
            run_speculative(job);
        }
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            --m_active_jobs;
        }
        m_idle_cv.notify_all();
    }
}

}  // namespace ov::intel_gpu::ocl::moe::lookahead
