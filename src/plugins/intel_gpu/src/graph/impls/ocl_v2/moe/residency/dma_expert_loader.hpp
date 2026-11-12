// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "../expert_weight_providers.hpp"
#include "expert_prefetch_scheduler.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/stream.hpp"

namespace ov::intel_gpu::ocl::moe::lookahead {

/// What an asynchronous load of one layer needs. Held (as PrefetchTarget::owner) by every queued or in-flight load, so
/// the provider and the device buffers outlive the load even if the network is torn down meanwhile.
struct LoadTarget {
    std::shared_ptr<OffloadExpertWeightProvider> provider;
    cldnn::moe_weights resident;  // copies of the buffer handles (shared ownership), taken after the provider was bound
};

/// Streams one expert from the OTD weights file into a slot of the layer's resident buffers on a dedicated DMA stream.
///
/// Uses only the generic stream/memory API (user events, barriers, memory::copy_from), so it is not tied to the OpenCL
/// runtime. The target's `owner` must be a std::shared_ptr<LoadTarget>.
///
///  * WAR (invariant I2): the DMA stream waits for the last kernel that read the slot before it overwrites it.
///  * RAW (invariant I1): the ready event of a load is signalled by the scheduler only after the copy has completed.
///  * A staging chunk holds one host payload per tensor of an expert, so the copies of one expert never alias each other.
///    The scheduler guarantees a chunk is not handed out again before the completion returned by begin_load() returned.
///  * Only the scheduler's worker thread enqueues on the DMA stream.
class DmaExpertLoader : public IExpertLoader {
public:
    DmaExpertLoader(cldnn::engine& engine, size_t staging_chunks);
    ~DmaExpertLoader() override;

    EventHandle make_ready_event() override;
    void signal(const EventHandle& event) noexcept override;
    Completion begin_load(const PrefetchTarget& target, uint32_t expert, size_t slot, const EventHandle& war_event, size_t staging_chunk) override;

    /// Stream on which the copies run (needed to create ready events that the compute stream can wait for).
    cldnn::stream& dma_stream() {
        return *m_dma_stream;
    }

private:
    cldnn::stream::ptr m_dma_stream;
    std::vector<std::vector<std::vector<uint8_t>>> m_staging;  // [chunk][tensor] payload, sized once in the constructor
};

}  // namespace ov::intel_gpu::ocl::moe::lookahead
