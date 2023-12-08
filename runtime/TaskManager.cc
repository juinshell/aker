// TaskManager.cpp
#include "TaskManager.h"

void TaskManager::addTask(Task& task) {
    tasks.push_back(&task);
}

void TaskManager::executeAllTasks() {
    for (auto& task : tasks) {
        task->executeTask();
    }
}