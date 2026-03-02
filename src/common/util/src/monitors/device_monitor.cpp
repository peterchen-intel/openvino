// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/device_monitor.hpp"

#include <algorithm>
#include <iostream>

#include "openvino/util/cpu_device_monitor.hpp"
#include "openvino/util/xpu_device_monitor.hpp"

namespace ov {
namespace util {

std::map<std::string, float> get_device_utilization(const std::string& device_name) {
    try {
        if (device_name.find("CPU") != std::string::npos) {
            return ov::util::CPUDeviceMonitor().get_utilization();
        } else {
            return ov::util::XPUDeviceMonitor(device_name).get_utilization();
        }
    } catch (...) {
        // Handle exceptions and return an empty map
        return {};
    }
}
}  // namespace util
}  // namespace ov