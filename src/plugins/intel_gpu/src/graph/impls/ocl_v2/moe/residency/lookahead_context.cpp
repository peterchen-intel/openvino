// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lookahead_context.hpp"

#include <algorithm>
#include <iostream>
#include <map>
#include <string>

#include "fully_connected_inst.h"
#include "impls/ocl_v2/primitive_ocl_base.hpp"
#include "intel_gpu/primitives/fully_connected.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "../moe_3gemm_base.hpp"
#include "intel_gpu/primitives/moe_router_fused.hpp"
#include "moe_3gemm_fused_inst.h"
#include "moe_router_fused_inst.h"
#include "openvino/util/log.hpp"

namespace ov::intel_gpu::ocl::moe::lookahead {

namespace {

// OPENVINO_WARN is compiled out unless the build has ENABLE_OPENVINO_DEBUG, but a layer that silently loses its
// prefetch is exactly what an operator who set GPU_MOE_LOOKAHEAD_PREFETCH needs to hear about. Reported once per layer.
void warn_layer_disabled(const std::string& layer_id, const std::string& reason) {
    const std::string msg = "[GPU] LES: layer " + layer_id + " is not prefetched: " + reason;
    OPENVINO_WARN(msg);
#ifndef ENABLE_OPENVINO_DEBUG
    std::cerr << msg << std::endl;
#endif
}

std::mutex g_registry_mutex;
std::unordered_map<uint32_t, std::weak_ptr<LookaheadContext>> g_registry;

}  // namespace

std::shared_ptr<LookaheadContext> LookaheadContext::acquire(cldnn::network& net) {
    std::lock_guard<std::mutex> lock(g_registry_mutex);
    for (auto it = g_registry.begin(); it != g_registry.end();) {
        it = it->second.expired() ? g_registry.erase(it) : std::next(it);
    }
    auto& slot = g_registry[net.get_id()];
    if (auto sp = slot.lock()) {
        // net ids are reused after a network is destroyed: only share the context with the very same network object.
        if (sp->network() == &net) {
            return sp;
        }
    }
    auto sp = std::shared_ptr<LookaheadContext>(new LookaheadContext(net, net.get_id()));
    slot = sp;
    return sp;
}

LookaheadContext::LookaheadContext(cldnn::network& net, uint32_t net_id) : m_network(&net), m_net_id(net_id) {
    m_loader = std::make_shared<DmaExpertLoader>(net.get_engine(), kStagingChunks);
    m_scheduler = std::make_unique<PrefetchScheduler>(m_loader, kStagingChunks);
}

LookaheadContext::~LookaheadContext() {
    // The scheduler joins its threads (draining every load that is still in flight) before the loader and the DMA stream
    // are destroyed, so no asynchronous copy can outlive the memory it writes.
    m_scheduler.reset();
    m_loader.reset();
}

size_t LookaheadContext::speculative_quota(size_t capacity, size_t top_k) {
    return std::max<size_t>(1, std::min(top_k, capacity / 4));
}

void LookaheadContext::build_topology(cldnn::network& net) {
    for (const auto& id : net.get_executed_primitive_ids()) {
        auto inst = net.get_primitive(id);
        if (inst && inst->type() == cldnn::moe_3gemm_fused_compressed::type_id()) {
            m_index[inst.get()] = m_order.size();
            m_order.push_back(inst.get());
        }
    }
}

namespace {

const char* wei_mode_name(int m) {
    switch (m) {
    case 0:
        return "u4";
    case 1:
        return "i4";
    case 2:
        return "u8";
    default:
        return "i8";
    }
}

}  // namespace

void LookaheadContext::analyze_router(const cldnn::primitive_inst& moe, Layer& layer) {
    layer.proxy_supported = false;
    auto fail = [&](std::string reason) {
        layer.unsupported_reason = std::move(reason);
    };

    // MoE input 2 (topk indices) is output 1 of the fused router, whose input 0 is the logits of the router FC.
    const auto& deps = moe.dependencies();
    if (deps.size() <= static_cast<size_t>(MOE3GemmInputIndex::TOPK_INDICES)) {
        return fail("unexpected MoE input count");
    }
    const auto* router = deps[static_cast<size_t>(MOE3GemmInputIndex::TOPK_INDICES)].first;
    if (router == nullptr || router->type() != cldnn::moe_router_fused::type_id()) {
        return fail("topk indices are not produced by moe_router_fused");
    }
    const auto router_desc = router->get_typed_desc<cldnn::moe_router_fused>();
    if (router_desc->_config.routing_type != cldnn::MoERouterFused::RoutingType::SOFTMAX) {
        // Sigmoid + bias changes the ranking relative to the raw logits, the proxy does not model it.
        return fail("router is not softmax");
    }
    const auto& router_deps = router->dependencies();
    if (router_deps.empty() || router_deps[0].first == nullptr) {
        return fail("router has no input");
    }
    const auto* fc = router_deps[0].first;
    if (fc->type() != cldnn::fully_connected::type_id()) {
        return fail("router logits are not produced by a fully connected primitive");
    }
    const auto fc_desc = fc->get_typed_desc<cldnn::fully_connected>();
    if (!fc_desc->compressed_weights || !fc_desc->decompression_scale.is_valid()) {
        return fail("router weights are not compressed");
    }
    if (fc_desc->bias.is_valid() || fc->has_fused_primitives()) {
        return fail("router FC has a bias or fused ops");
    }
    if (fc_desc->decompression_zero_point_scalar.has_value()) {
        return fail("router FC uses a scalar zero point");
    }
    if (fc_desc->dynamic_quantized_activation || fc_desc->hadamard_block != 0) {
        return fail("router FC uses activation quantization or a Hadamard transform");
    }

    // FC dependencies: [0] input, then weights, [bias], scale, [zp] (see fully_connected::get_dependencies_map).
    size_t idx = fc_desc->input.size();
    const auto weights = fc->dep_memory_ptr(idx++);
    if (fc_desc->bias.is_valid()) {
        ++idx;
    }
    const auto scales = fc->dep_memory_ptr(idx++);
    cldnn::memory::ptr zps;
    if (fc_desc->decompression_zero_point.is_valid()) {
        zps = fc->dep_memory_ptr(idx++);
    }
    if (!weights || !scales) {
        return fail("router weight or scale buffer is not available");
    }

    const auto wl = weights->get_layout();
    const auto sl = scales->get_layout();
    const size_t E = layer.num_experts;
    const size_t H = layer.hidden_size;
    if (!cldnn::format::is_default_format(wl.format)) {
        return fail("router weights are not in the plain layout");
    }
    if (sl.data_type != ov::element::f16) {
        return fail("router scale is not f16");
    }
    int wei_mode = -1;
    switch (wl.data_type) {
    case ov::element::u4:
        wei_mode = 0;
        break;
    case ov::element::i4:
        wei_mode = 1;
        break;
    case ov::element::u8:
        wei_mode = 2;
        break;
    case ov::element::i8:
        wei_mode = 3;
        break;
    default:
        return fail("router weight type is not u4/i4/u8/i8");
    }
    if (wl.count() != E * H) {
        return fail("router weight element count does not match experts x hidden");
    }
    if (sl.count() == 0 || sl.count() % E != 0 || H % (sl.count() / E) != 0) {
        return fail("router scale shape does not match the weights");
    }
    const size_t groups = sl.count() / E;
    const size_t group_size = H / groups;

    int zp_mode = 0;
    if (zps) {
        const auto zl = zps->get_layout();
        if (zl.count() != sl.count()) {
            return fail("router zero point count differs from the scale count (scalar or per-tensor zero point)");
        }
        if (zl.data_type == ov::element::u4) {
            zp_mode = 1;
        } else if (zl.data_type == ov::element::u8) {
            zp_mode = 2;
        } else {
            return fail("router zero point type is not u4/u8");
        }
    }
    if (zp_mode == 1 && (sl.count() % 2) != 0) {
        return fail("odd u4 zero point count");
    }
    if (wei_mode <= 1 && ((E * H) % 2) != 0) {
        return fail("odd 4-bit weight count");
    }

    layer.router.weights = weights;
    layer.router.scales = scales;
    layer.router.zero_points = zps;
    layer.router.wei_mode = wei_mode;
    layer.router.zp_mode = zp_mode;
    layer.router.group_size = static_cast<int>(group_size);
    layer.router.scale_transposed = fc_desc->transpose_b_scale ? 1 : 0;
    layer.proxy_supported = true;
    layer.unsupported_reason.clear();
    GPU_DEBUG_TRACE_DETAIL << "LES layer " << layer.index << ": proxy router " << wei_mode_name(wei_mode) << " group_size=" << group_size
                           << " zp_mode=" << zp_mode << " scale_transposed=" << layer.router.scale_transposed << std::endl;
}

Layer& LookaheadContext::register_layer(cldnn::network& net,
                                        const cldnn::primitive_inst& moe,
                                        std::shared_ptr<OffloadExpertWeightProvider> provider,
                                        const cldnn::moe_weights& resident) {
    std::call_once(m_topology_once, [&] {
        build_topology(net);
    });
    std::lock_guard<std::mutex> lock(m_mutex);
    auto& slot = m_layers[&moe];
    if (!slot) {
        slot = std::make_unique<Layer>();
        slot->inst = &moe;
        const auto desc = moe.get_typed_desc<cldnn::moe_3gemm_fused_compressed>();
        slot->num_experts = desc->_config.num_expert;
        slot->hidden_size = desc->_config.hidden_size;
        slot->top_k = desc->_config.top_k;
        auto it = m_index.find(&moe);
        slot->index = it == m_index.end() ? static_cast<size_t>(-1) : it->second;
        analyze_router(moe, *slot);
        if (!slot->proxy_supported) {
            warn_layer_disabled(moe.id(), slot->unsupported_reason);
        }
    }
    if (!slot->load_target || slot->provider != provider) {
        slot->load_target = std::make_shared<LoadTarget>(LoadTarget{provider, resident});
    }
    slot->provider = std::move(provider);
    return *slot;
}

Layer* LookaheadContext::find(const cldnn::primitive_inst* moe) {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_layers.find(moe);
    return it == m_layers.end() ? nullptr : it->second.get();
}

Layer* LookaheadContext::next_of(const cldnn::primitive_inst* moe) {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto idx = m_index.find(moe);
    if (idx == m_index.end() || idx->second + 1 >= m_order.size()) {
        return nullptr;
    }
    auto next = m_layers.find(m_order[idx->second + 1]);
    if (next == m_layers.end()) {
        return nullptr;  // not executed yet (first token): its provider and router are not known
    }
    Layer* layer = next->second.get();
    if (!layer->proxy_supported || !layer->provider || !layer->load_target) {
        return nullptr;
    }
    return layer;
}

cldnn::memory::ptr LookaheadContext::proxy_logits(cldnn::engine& engine, size_t num_experts) {
    std::lock_guard<std::mutex> lock(m_mutex);
    const size_t bytes = num_experts * sizeof(float);
    if (!m_logits || m_logits->size() < bytes) {
        m_logits = engine.allocate_memory(cldnn::layout({1, 1, 1, static_cast<ov::Dimension::value_type>(bytes)}, ov::element::i8, cldnn::format::bfyx),
                                          cldnn::allocation_type::usm_device,
                                          false);
    }
    return m_logits;
}

cldnn::memory::ptr LookaheadContext::proxy_ids(cldnn::engine& engine, size_t top_k) {
    std::lock_guard<std::mutex> lock(m_mutex);
    const size_t bytes = top_k * sizeof(int32_t);
    if (!m_ids || m_ids->size() < bytes) {
        m_ids = engine.allocate_memory(cldnn::layout({1, 1, 1, static_cast<ov::Dimension::value_type>(bytes)}, ov::element::i8, cldnn::format::bfyx),
                                       cldnn::allocation_type::usm_host,
                                       false);
    }
    return m_ids;
}

}  // namespace ov::intel_gpu::ocl::moe::lookahead
