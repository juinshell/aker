/*** 
 * @Author: diagonal
 * @Date: 2023-12-09 11:59:08
 * @LastEditors: diagonal
 * @LastEditTime: 2023-12-09 12:41:48
 * @FilePath: /tacker/runtime/ModuleCenter.cu
 * @Description: 
 * @happy coding, happy life!
 * @Copyright (c) 2023 by jxdeng, All Rights Reserved. 
 */
#include "ModuleCenter.h"
#include "util.h"
#include "Logger.h"

extern Logger logger;


bool ModuleCenter::registerModule(const std::string& moduleName, const std::string& cubinPath) {
    ModuleInfo moduleInfo;
    
    // 加载 cubin 文件并初始化函数映射
    if (!loadModule(cubinPath, moduleInfo)) {
        return false;
    }

    // 注册 Module
    modules[moduleName] = moduleInfo;
    return true;
}

CUfunction ModuleCenter::getFunction(const std::string& moduleName, const std::string& functionName) {
    // 查找指定 Module
    auto moduleIt = modules.find(moduleName);
    if (moduleIt == modules.end()) {
        // 没找到 Module
        logger.ERROR("can't find module: " + moduleName);
        return nullptr;
    }

    // 查找指定函数
    auto functionIt = moduleIt->second.functionMap.find(functionName);
    if (functionIt == moduleIt->second.functionMap.end()) {
        // 没找到函数, 尝试从module中load
        CUfunction function;
        CU_SAFE_CALL(cuModuleGetFunction(&function, moduleIt->second.module, functionName.c_str()));
        moduleIt->second.functionMap[functionName] = function;
        // logger.INFO("success load function: " + functionName);
        return function;
    }

    // logger.INFO("success find function: " + functionName);
    return functionIt->second;
}

bool ModuleCenter::loadModule(const std::string& cubinPath, ModuleInfo& moduleInfo) {
    // 创建 CUmodule
    CU_SAFE_CALL(cuModuleLoad(&moduleInfo.module, cubinPath.c_str()));

    logger.INFO("success load module: " + cubinPath);

    return true;
}