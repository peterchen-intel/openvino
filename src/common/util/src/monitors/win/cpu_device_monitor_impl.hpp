// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <map>

#include "openvino/util/idevice_monitor.hpp"

#define NOMINMAX
#include <windows.h>

#include <chrono>
#include <filesystem>
#include <string>
#include <system_error>
#include <thread>

namespace ov::util {
class CPUDeviceMonitorImpl : public IDeviceMonitorImpl {
public:
    CPUDeviceMonitorImpl();
    constexpr int query_number_of_cores() {
        // Query the number of logical processors
        // Only focuses on the total usage, so we can use _Total counter
        return 0;
    }
    std::map<std::string, float> get_utilization() override;
};

}  // namespace ov::util