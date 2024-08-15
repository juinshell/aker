/*** 
 * @Author: diagonal
 * @Date: 2023-12-09 11:56:33
 * @LastEditors: diagonal
 * @LastEditTime: 2023-12-09 12:19:04
 * @FilePath: /tacker/runtime/ModuleCenter.h
 * @Description: 
 * @happy coding, happy life!
 * @Copyright (c) 2023 by jxdeng, All Rights Reserved. 
 */
#pragma once
#include "util.h"
#include <vector>
#include <map>

class ModuleCenter {
public:

    ModuleCenter() {};
    ~ModuleCenter() {
        for (auto& module : modules) {
            cuModuleUnload(module.second.module);
        }
        // clear two map
        for (auto& module : modules) {
            module.second.functionMap.clear();
        }
        modules.clear();
    };
    // 注册一个 Module
    bool registerModule(const std::string& moduleName, const std::string& cubinPath);

    // 获取指定 Module 中的函数
    CUfunction getFunction(const std::string& moduleName, const std::string& functionName);

private:
    struct ModuleInfo {
        CUmodule module;
        std::map<std::string, CUfunction> functionMap;
    };

    std::map<std::string, ModuleInfo> modules;

    // 加载 cubin 文件并初始化函数映射
    bool loadModule(const std::string& cubinPath, ModuleInfo& moduleInfo);
};