// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpu_device_monitor_impl.hpp"
//#include <nlohmann/json.hpp>
#include "json.hpp"  // Include the JSON library
#include <iostream>

using Json = nlohmann::json;

#include <ctime>
#include <fstream>
#include <iostream>
#include <string>

namespace ov {
namespace util {
CPUDeviceMonitorImpl::CPUDeviceMonitorImpl() {

}

std::map<std::string, float> CPUDeviceMonitorImpl::get_utilization() {
    HANDLE hPipe = CreateFile(TEXT("\\\\.\\pipe\\XpuInfoServicePipe"), GENERIC_READ, 0, NULL, OPEN_EXISTING, 0, NULL);

    if (hPipe == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("CreateFile() failed. Is XPU service running?");
        return {};
    }
    
    char buffer[2048];
    DWORD bytesRead;

    if (!ReadFile(hPipe, buffer, sizeof(buffer) - 1, &bytesRead, NULL)) {
        throw std::runtime_error("ReadFile() failed.");
        return {};
    }

    buffer[bytesRead] = '\0';
    CloseHandle(hPipe);

    std::cout << "[CPUDeviceMonitor] XPU Info:" << buffer << std::endl;

    // Parse the JSON string
    Json parsed = Json::parse(buffer);

    // Extract the first array (core data)
    auto coreData = parsed[0];
    std::vector<float> core_Utilization;
    std::vector<float> atom_Utilization;

    float coreSum = 0.0, atomSum = 0.0;
    int coreCount = 0, atomCount = 0;

    for (const auto& entry : coreData) {
        std::string type = entry["Type"];
        float utilization = entry["Utilization"];

        if (type == "Core") {
            coreSum += utilization;
            core_Utilization.push_back(utilization);
            coreCount++;
        } else if (type == "Atom") {
            atomSum += utilization;
            atom_Utilization.push_back(utilization);
            atomCount++;
        }
    }

    std::map<std::string, float> cpus_utilization;
    int n_cores = query_number_of_cores(); /*Presently, this is returning 0*/
    if (n_cores == 0) {
        float avgAll = (coreSum + atomSum) / (coreCount + atomCount);
        cpus_utilization["CPU"] = avgAll;
    } else {
        for (int index = 0; index < core_Utilization.size(); index++) {
            cpus_utilization[std::to_string(index)] = core_Utilization.at(index);
        }    
        for (int index = 0; index < atom_Utilization.size(); index++) {
            cpus_utilization[std::to_string(index)] = atom_Utilization.at(index);
        }
    }

    std::cout << "CPU utilization:" << cpus_utilization["CPU"] << std::endl;

    return cpus_utilization;
}
}  // namespace util
}  // namespace ov