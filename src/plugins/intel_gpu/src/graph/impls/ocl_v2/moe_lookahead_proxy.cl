// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// LES v1 lookahead proxy: predicts the experts of the next MoE layer from the hidden state of the current one.
//
// The result is only a prefetch hint. The real router of the next layer always decides what is computed, so a wrong
// prediction costs overlap, never correctness.
//
// Two kernels, both single dispatch, both on the compute queue:
//   proxy_fc   logits[e] = sum_h hidden[h] * dequant(W[e, h]), one work-group per expert
//   proxy_topk top-K expert ids by logit (descending, ties to the lower id), written to a host-visible buffer
//
// Softmax is monotonic, so the ranking of the logits equals the ranking of the router probabilities and the softmax is
// not computed. The weights are read in the original (plain row-major) layout of the router FC constant.
// Half values are read with vload_half so the kernel does not depend on cl_khr_fp16.
//
// Jit constants: NUM_EXPERTS, HIDDEN_SIZE, TOP_K, PROXY_LWS (work-group size, power of two).
// Runtime scalars of proxy_fc (they differ between layers, so they are arguments and not jit constants):
//   wei_mode   0 u4, 1 i4, 2 u8, 3 i8
//   zp_mode    0 none (zero point is 0), 1 u4 per (expert, group), 2 u8 per (expert, group)
//   group_size input channels per scale / zero point
//   scale_transposed  1: scale / zp are [expert, group], 0: [group, expert]

#if PROXY_FC_ENABLE

KERNEL(proxy_fc)(const __global half* hidden,
                 const __global uchar* weights,
                 const __global half* scales,
                 const __global uchar* zps,
                 __global float* logits,
                 const int wei_mode,
                 const int zp_mode,
                 const int group_size,
                 const int scale_transposed) {
    const uint expert = get_group_id(0);
    const uint lid = get_local_id(0);
    const uint num_groups = HIDDEN_SIZE / (uint)group_size;

    float sum = 0.0f;
    for (uint h = lid; h < HIDDEN_SIZE; h += PROXY_LWS) {
        const uint widx = expert * HIDDEN_SIZE + h;
        int w;
        if (wei_mode <= 1) {
            const uint byte = weights[widx >> 1];
            const int nib = (widx & 1) ? (int)(byte >> 4) : (int)(byte & 0x0F);
            w = (wei_mode == 1 && nib >= 8) ? nib - 16 : nib;
        } else if (wei_mode == 2) {
            w = (int)weights[widx];
        } else {
            w = (int)((char)weights[widx]);
        }

        const uint g = h / (uint)group_size;
        const uint sidx = scale_transposed ? (expert * num_groups + g) : (g * NUM_EXPERTS + expert);
        int zp = 0;
        if (zp_mode == 1) {
            const uint byte = zps[sidx >> 1];
            zp = (sidx & 1) ? (int)(byte >> 4) : (int)(byte & 0x0F);
        } else if (zp_mode == 2) {
            zp = (int)zps[sidx];
        }
        sum = fma(((float)(w - zp)) * vload_half(sidx, scales), vload_half(h, hidden), sum);
    }

    __local float partial[PROXY_LWS];
    partial[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = PROXY_LWS / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            partial[lid] += partial[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        logits[expert] = partial[0];
    }
}

#elif PROXY_TOPK_ENABLE

KERNEL(proxy_topk)(const __global float* logits, __global int* out_ids) {
    const uint lid = get_local_id(0);

    __local float l[NUM_EXPERTS];
    for (uint i = lid; i < NUM_EXPERTS; i += PROXY_LWS) {
        l[i] = logits[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // rank(e) = number of experts that sort before e. Experts with rank < TOP_K are the top-K, and rank is their slot.
    for (uint e = lid; e < NUM_EXPERTS; e += PROXY_LWS) {
        const float v = l[e];
        uint rank = 0;
        for (uint j = 0; j < NUM_EXPERTS; ++j) {
            const float o = l[j];
            rank += (o > v || (o == v && j < e)) ? 1u : 0u;
        }
        if (rank < TOP_K) {
            out_ids[rank] = (int)e;
        }
    }
}

#endif
