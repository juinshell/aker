'''
Author: diagonal
Date: 2023-11-15 22:45:42
LastEditors: diagonal
LastEditTime: 2023-11-16 14:19:08
FilePath: /tacker/scripts/test.py
Description: 
happy coding, happy life!
Copyright (c) 2023 by jxdeng, All Rights Reserved. 
'''
import re

def extract_kernel_signature(kernel_code):
    # 使用正则表达式匹配 CUDA kernel 函数签名
    pattern = re.compile(r'__global__\s+void\s+(\w+)\s*\(([^)]*)\)')
    match = pattern.search(kernel_code)

    if match:
        # 如果匹配成功，返回函数签名的两个组成部分：函数名和参数
        function_name = match.group(1)
        function_params = match.group(2).strip()
        return function_name, function_params
    else:
        # 如果匹配失败，输出错误信息，退出进程
        print("Failed to extract kernel signature.")
        exit(1)


def generate_mixed_kernel(kernel_info):
    # 从第一个 kernel 获取参数
    first_kernel_signature = kernel_info["signatures"][0]
    first_func_name, first_func_full_params = extract_kernel_signature(first_kernel_signature)
    first_func_name = first_func_name.split('_')[-1].replace(' ', '')
    print("first_func_name: ", first_func_name, end='\n\n')
    print("first_func_full_params: ", first_func_full_params, end='\n\n')
    params_list = first_func_full_params.split(', ')
    print("params_list: ", params_list, end='\n\n')
    params_names_list = [param.split()[-1] for param in params_list]
    params_names_list = params_names_list[:-1]
    for i in range(len(params_names_list)):
        params_names_list[i] = first_func_name + '0' + params_names_list[i]
    print("first params_names_list: ", params_names_list, end='\n\n')

    # 生成混合 kernel 的代码
    mixed_kernel_code = f"""__global__ void mixed_$NAME_kernel({", ".join(params_names_list)}) """ + '{\n'
    thread_idx_threshold = 0
    for i, kernel_signature in enumerate(kernel_info["signatures"]):
        if (i == 0):
            continue
        parameters = kernel_signature.split("(")[1].split(")")[0].split(", ")

        print("kernel_params: ", parameters)
        mixed_kernel_code += f"    if (threadIdx.x < {thread_idx_threshold}) {{\n"
        mixed_kernel_code += f"        {kernel_signature.split('(')[1]}(\n"
        mixed_kernel_code += f"            // Pass other parameters here\n"
        mixed_kernel_code += f"        );\n"
        mixed_kernel_code += f"    }}\n"
        mixed_kernel_code += f"    else "  # Add 'else' for the next condition

    mixed_kernel_code += """
    {
        // Handle else case if needed
    }
}
"""

    return mixed_kernel_code

# 示例使用：三个 kernel，每个 kernel 有不同的 block dimension
kernel_info = {
    "signatures": [
        "__global__ void general_ptb_cp(int numatoms, float gridspacing, float * energygrid, int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z, int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base)",
        "__global__ void another_kernel(int numatoms, float gridspacing, float * energygrid, int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z, int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base)",
        # Add more kernel signatures as needed
    ],
}

mixed_kernel_code = generate_mixed_kernel(kernel_info)

print("\nGenerated Mixed Kernel Code:")
print(mixed_kernel_code)
