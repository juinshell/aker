// TaskManager.h
#pragma once

#include "Task.h"
#include <vector>

class TaskManager {
public:
    TaskManager() {};
    void addTask(Task& task);
    void executeAllTasks();
    ~TaskManager() {};

private:
    std::vector<Task*> tasks;
};