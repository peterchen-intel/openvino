// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file cpu_streams_calculation.hpp
 * @brief A header file for CPU streams calulation implementation.
 */

#pragma once

#include <memory>
#include <vector>

#include "graph.h"

namespace ov {
namespace intel_cpu {
/**
 * @brief      Generate streams information table according to processors type table
 * @param[in]  input_streams is target streams set by user via NUM_STREAMS or hints.
 *               - input "0" mean function generate the optimal number of streams
 *               - LATENCY hint equals 1 stream.
 * @param[in]  input_threads is max threads set by user via INFERNECE_NUM_THREADS.
 *               - input "0" mean function can use all resource in proc_type_table
 *               - When user limit max threads, streams in output cannot be more than max threads
 * @param[in]  model_prefer_threads is preferred threads per stream based on model generated in previous function
 *               - input "0" mean function generate the optimal threads per stream based on platform
 * @param[in]  proc_type_table candidate processors available at this time
 *               - candidate processors have benn updated based on properties like "Ecore only" in previous function
 * @return     summary table of streams info will be used by StreamsExecutor
 */
std::vector<std::vector<int>> get_streams_info_table(const int input_streams,
                                                     const int input_threads,
                                                     const int model_prefer_threads,
                                                     const std::vector<std::vector<int>> proc_type_table);
/**
 * @brief      Get model_prefer_threads
 * @param[in]  num_streams is target streams set by user via NUM_STREAMS or hints.
 *               - input "0" mean function generate the optimal number of streams
 *               - LATENCY hint equals 1 stream.
 * @param[in]  proc_type_table candidate processors available at this time
 *               - candidate processors have benn updated based on properties like "Ecore only" in previous function
 * @param[in]  ngraphFunc ngraph function
 * @return     model_prefer_threads "0" means generating the optimal threads per stream based on platform
 */
int get_model_prefer_threads(const int num_streams,
                             const std::vector<std::vector<int>> proc_type_table,
                             const std::shared_ptr<ngraph::Function>& ngraphFunc,
                             const InferenceEngine::IStreamsExecutor::Config streamExecutorConfig);

struct StreamCfg {
    int num_streams;               // Number of streams
    int num_threads;               // Number of threads
    int big_core_streams;          // Number of streams in Performance-core(big core)
    int big_core_logic_streams;    // Number of streams in Performance logical core(big core)
    int small_core_streams;        // Number of streams in Efficient-core(small core)
    int threads_per_stream_big;    // Threads per stream in big cores
    int threads_per_stream_small;  // Threads per stream in small cores
    int small_core_offset;
};

/**
 * @brief      Parse streams table to StreamCfg
 * @param[in]  streams_table summary table of streams info will be used by StreamsExecutor
 * @return     Streams info
 */
StreamCfg parse_streams_table(std::vector<std::vector<int>> streams_table);

std::pair<std::string, StreamCfg> get_num_streams(const int streams,
                                                  const std::shared_ptr<ngraph::Function>& ngraphFunc,
                                                  const InferenceEngine::IStreamsExecutor::Config streamExecutorConfig);

}  // namespace intel_cpu
}  // namespace ov

