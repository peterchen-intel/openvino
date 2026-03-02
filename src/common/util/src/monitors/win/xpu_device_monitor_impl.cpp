// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "xpu_device_monitor_impl.hpp"

#include "json.hpp"  // Include the JSON library

using Json = nlohmann::json;

namespace ov {
namespace util {
XPUDeviceMonitorImpl::XPUDeviceMonitorImpl(const std::string& device_name) {
    m_device_name = device_name;
}

std::map<std::string, float> XPUDeviceMonitorImpl::get_utilization() {
    HANDLE hPipe = CreateFile(TEXT("\\\\.\\pipe\\XpuInfoServicePipe"), GENERIC_READ, 0, NULL, OPEN_EXISTING, 0, NULL);

    if (hPipe == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("CreateFile() failed. Is XPU service running?");
    }

    char buffer[2048];
    DWORD bytesRead;

    if (!ReadFile(hPipe, buffer, sizeof(buffer) - 1, &bytesRead, NULL)) {
        throw std::runtime_error("ReadFile() failed.");
    }

    buffer[bytesRead] = '\0';
    CloseHandle(hPipe);

    // Parse the JSON string
    Json parsed = Json::parse(buffer);

    std::map<std::string, float> utilization_map;

    if (m_device_name.find("GPU") != std::string::npos) {
        utilization_map[m_device_name] = static_cast<float>(parsed[1]);
    }
    else if (m_device_name.find("NPU") != std::string::npos) {
        utilization_map[m_device_name] = static_cast<float>(parsed[2]);
    }
    return utilization_map;
}
}  // namespace util
}  // namespace ov