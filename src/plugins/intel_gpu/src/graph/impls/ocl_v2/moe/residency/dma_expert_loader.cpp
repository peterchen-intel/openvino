// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dma_expert_loader.hpp"

#include <utility>

// primitive_ocl_base.hpp is the established prelude of the OTD sources: moe_otd_runtime.hpp is not self-contained.
#include "../expert_weight_providers.hpp"
#include "impls/ocl_v2/primitive_ocl_base.hpp"
#include "intel_gpu/runtime/event.hpp"
#include "intel_gpu/runtime/execution_config.hpp"
#include "openvino/core/except.hpp"

namespace ov::intel_gpu::ocl::moe::lookahead {

namespace {
constexpr size_t kTensorCount = cldnn::moe_3gemm_fused_compressed::serialized_weight_offset_count;
}

DmaExpertLoader::DmaExpertLoader(cldnn::engine& engine, size_t staging_chunks) {
    // An in-order queue of its own: copies of different experts stay ordered, and the only cross-queue dependencies are
    // the explicit events (WAR before a copy, ready after it).
    cldnn::ExecutionConfig config;
    config.set_property(ov::intel_gpu::queue_type(cldnn::QueueTypes::in_order));
    m_dma_stream = engine.create_stream(config);
    m_staging.resize(staging_chunks);
    for (auto& chunk : m_staging) {
        chunk.resize(kTensorCount);
    }
}

DmaExpertLoader::~DmaExpertLoader() {
    // The scheduler is destroyed first and has waited for every copy, this only drains what is left on the queue.
    try {
        m_dma_stream->finish();
    } catch (...) {
    }
}

EventHandle DmaExpertLoader::make_ready_event() {
    return m_dma_stream->create_user_event(false);
}

void DmaExpertLoader::signal(const EventHandle& event) noexcept {
    try {
        if (event) {
            event->set();
        }
    } catch (...) {
        // Nothing sensible can be done here and the contract is not to throw.
    }
}

IExpertLoader::Completion DmaExpertLoader::begin_load(const PrefetchTarget& target,
                                                      uint32_t expert,
                                                      size_t slot,
                                                      const EventHandle& war_event,
                                                      size_t staging_chunk) {
    auto load_target = std::static_pointer_cast<LoadTarget>(target.owner);
    OPENVINO_ASSERT(load_target && load_target->provider, "LES: prefetch target without a weight provider");
    OPENVINO_ASSERT(staging_chunk < m_staging.size(), "LES: staging chunk index out of range");

    auto& provider = load_target->provider;
    cldnn::moe_weights resident = load_target->resident;
    auto& reader = provider->weight_reader();
    const auto plans = moe_otd::plan_expert_tensors(provider->config(), provider->weight_bin_offsets(), reader.file_size(), resident, expert, slot);

    // I2 (write after read): the copies must not start before the last kernel that read the previous content of the slot.
    if (war_event) {
        m_dma_stream->enqueue_barrier({war_event});
    }

    auto& payloads = m_staging[staging_chunk];
    std::vector<cldnn::event::ptr> copies;
    copies.reserve(plans.size());
    try {
        for (const auto& plan : plans) {
            auto& payload = payloads[plan.offset_pos];
            payload.resize(plan.per_expert_size);
            reader.read(reinterpret_cast<char*>(payload.data()), plan.per_expert_size, plan.src_offset);
            moe_otd::maybe_transpose_scale_zp(provider->config(), plan.offset_pos, plan.dst->get_layout(), payload, plan.per_expert_size);
            OPENVINO_ASSERT(payload.size() == plan.per_expert_size, "LES: payload size changed by the layout conversion");
            auto ev = plan.dst->copy_from(*m_dma_stream, payload.data(), 0, plan.dst_offset, plan.per_expert_size, false);
            if (ev) {
                copies.push_back(std::move(ev));
            }
        }
    } catch (...) {
        // Copies that were already enqueued still read the payloads: let them finish before the chunk is reused.
        for (auto& c : copies) {
            try {
                c->wait();
            } catch (...) {
            }
        }
        throw;
    }
    // Events that another queue waits on make no progress until the queue that owns them is flushed.
    m_dma_stream->flush();

    return [copies = std::move(copies)]() {
        for (const auto& c : copies) {
            c->wait();
        }
    };
}

}  // namespace ov::intel_gpu::ocl::moe::lookahead
