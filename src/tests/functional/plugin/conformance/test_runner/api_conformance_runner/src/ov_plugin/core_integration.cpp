// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/core_integration_sw.hpp"
#include "behavior/ov_plugin/query_model.hpp"
#include "openvino/runtime/core.hpp"
#include "ov_api_conformance_helpers.hpp"

using namespace ov::test::behavior;
using namespace ov::test::conformance;

namespace {
//
// OV Class Common tests with <pluginName, deviceName params>
//

INSTANTIATE_TEST_SUITE_P(ov_plugin,
                         OVClassModelOptionalTestP,
                         ::testing::Values(ov::test::utils::target_device));

// OV Class Query network

INSTANTIATE_TEST_SUITE_P(ov_plugin_mandatory,
                         OVClassQueryModelTest,
                         ::testing::Values(ov::test::utils::target_device));
}  // namespace
