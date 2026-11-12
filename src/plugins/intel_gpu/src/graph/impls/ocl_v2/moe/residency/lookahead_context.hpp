// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "../expert_weight_providers.hpp"
#include "dma_expert_loader.hpp"
#include "expert_prefetch_scheduler.hpp"
#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/runtime/memory.hpp"

namespace ov::intel_gpu::ocl::moe::lookahead {

/// Everything the proxy kernel needs to rank the experts of one layer, derived from the layer's router FC.
struct RouterWeights {
    cldnn::memory::ptr weights;
    cldnn::memory::ptr scales;
    cldnn::memory::ptr zero_points;  // may be null (no zero point)
    int wei_mode = 0;                // 0 u4, 1 i4, 2 u8, 3 i8 (see moe_lookahead_proxy.cl)
    int zp_mode = 0;                 // 0 none, 1 u4, 2 u8
    int group_size = 0;
    int scale_transposed = 1;        // 1: [expert, group], 0: [group, expert]
};

struct Layer {
    const cldnn::primitive_inst* inst = nullptr;
    std::shared_ptr<OffloadExpertWeightProvider> provider;
    std::shared_ptr<LoadTarget> load_target;  // keeps provider and the resident device buffers alive for in-flight loads
    size_t index = 0;  // position among the MoE layers in execution order
    size_t num_experts = 0;
    size_t hidden_size = 0;
    size_t top_k = 0;
    bool proxy_supported = false;  // this layer's router can be evaluated by the proxy kernel
    RouterWeights router;
    std::string unsupported_reason;
};

/// Network-level state of LES v1: the order of the MoE layers, per layer the data the proxy needs, and the shared
/// asynchronous loader. It lives exactly as long as the MoE implementations of the network that hold it, so that the
/// network needs no knowledge of LES (no change to network.cpp / primitive_inst.h).
class LookaheadContext {
public:
    /// Context of `net`, created on first use. Held weakly by the registry, strongly by the MoE implementations.
    static std::shared_ptr<LookaheadContext> acquire(cldnn::network& net);

    ~LookaheadContext();

    uint32_t net_id() const {
        return m_net_id;
    }
    const cldnn::network* network() const {
        return m_network;
    }

    /// Registers (or refreshes) the layer of `moe`. Must be called from the inference thread after the provider was
    /// bound to the resident buffers. Resolves the router weights and the position in the layer order.
    Layer& register_layer(cldnn::network& net,
                          const cldnn::primitive_inst& moe,
                          std::shared_ptr<OffloadExpertWeightProvider> provider,
                          const cldnn::moe_weights& resident);

    Layer* find(const cldnn::primitive_inst* moe);
    /// Next MoE layer of `moe` in execution order if it is registered, its proxy is supported and its shape matches the
    /// shape `moe` was compiled for. nullptr otherwise (last layer, first token, unsupported router, ...).
    Layer* next_of(const cldnn::primitive_inst* moe);

    PrefetchScheduler& scheduler() {
        return *m_scheduler;
    }

    /// Scratch buffers of the proxy kernels, sized for the largest layer. Allocated once, never grown per token.
    cldnn::memory::ptr proxy_logits(cldnn::engine& engine, size_t num_experts);
    cldnn::memory::ptr proxy_ids(cldnn::engine& engine, size_t top_k);

    /// Largest number of loads the speculative path may keep in flight for a cache of `capacity` slots.
    static size_t speculative_quota(size_t capacity, size_t top_k);

    /// Staging chunks of the DMA loader. Small on purpose: it bounds both host memory and the time a demand load can
    /// wait behind speculative copies (head-of-line blocking).
    static constexpr size_t kStagingChunks = 3;

private:
    LookaheadContext(cldnn::network& net, uint32_t net_id);
    void build_topology(cldnn::network& net);
    void analyze_router(const cldnn::primitive_inst& moe, Layer& layer);

    cldnn::network* m_network;
    uint32_t m_net_id;
    std::mutex m_mutex;
    std::once_flag m_topology_once;
    std::vector<const cldnn::primitive_inst*> m_order;  // MoE instances in execution order
    std::unordered_map<const cldnn::primitive_inst*, size_t> m_index;
    std::unordered_map<const cldnn::primitive_inst*, std::unique_ptr<Layer>> m_layers;
    std::shared_ptr<DmaExpertLoader> m_loader;
    std::unique_ptr<PrefetchScheduler> m_scheduler;
    cldnn::memory::ptr m_logits;
    cldnn::memory::ptr m_ids;
};

}  // namespace ov::intel_gpu::ocl::moe::lookahead
