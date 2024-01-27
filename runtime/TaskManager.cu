// TaskManager.cpp
#include "TaskManager.h"
#include "Logger.h"

extern Logger logger;

void TaskManager::addTask(Task& task) {
    tasks.push_back(&task);
}

void TaskManager::executeAllTasks(ExecutionMode mode) {
    for (auto& task : tasks) {
        task->executeTask(mode);
    }
}

TaskManager::~TaskManager() {
    // logger.INFO("TaskManager is destroyed!");
}

