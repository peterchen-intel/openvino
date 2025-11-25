// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <map>

#include "openvino/util/idevice_monitor.hpp"

#define NOMINMAX
#include <pdh.h>
#include <pdhmsg.h>
#include <windows.h>

#include <chrono>
#include <filesystem>
#include <string>
#include <system_error>
#include <thread>

namespace ov::util {
class XPUDeviceMonitorImpl : public IDeviceMonitorImpl {
public:
    XPUDeviceMonitorImpl(const std::string& device_name);
    std::map<std::string, float> get_utilization() override;

private:
    std::string m_device_name;
};
}  // namespace ov::util