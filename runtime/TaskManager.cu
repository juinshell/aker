// TaskManager.cpp
#include "TaskManager.h"
#include "Logger.h"

extern Logger logger;

void TaskManager::addTask(Task& task) {
    tasks.push_back(&task);
}

void TaskManager::executeAllTasks() {
    for (auto& task : tasks) {
        task->executeTask();
    }
}

TaskManager::~TaskManager() {
    // logger.INFO("TaskManager is destroyed!");
}

