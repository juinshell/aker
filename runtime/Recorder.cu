#include "Recorder.h"
#include <fstream>
#include <iomanip>
#include <iostream>

void Recorder::text() {
    std::string filename = "recorder_output.txt";
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "cannot open file: " << filename << std::endl;
        return;
    }

    for (const auto& task_pair : task_time_map) {
        int taskId = task_pair.first;
        const auto& times = task_pair.second;
        const auto& kernelIds = task_kernel_map[taskId];
        const std::string& taskName = task_name_map[taskId];

        file << "taskID: " << taskId << ", taskName: " << taskName << "\n";

        float totalTaskTime = 0.0;
        for (size_t i = 0; i < times.size(); ++i) {
            int kernelId = kernelIds[i];
            float time = times[i];
            totalTaskTime += time;
            const std::string& kernelName = kernel_name_map[kernelId];

            file << "    kernelID: " << kernelId << ", kernelName: " << kernelName
                 << ", execTime: " << time << "ms\n";
        }

        file << "    task running time: " << totalTaskTime << "ms\n\n";
    }
}