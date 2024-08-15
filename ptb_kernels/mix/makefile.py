import os
import re

def extract_compile_params(line):
    # 使用正则表达式提取编译参数
    match = re.search(r'nvcc -o [^\s]+ ([^\n]+) \S+\.cu', line)
    if match:
        return match.group(1).strip()
    return None

def generate_makefile_rules(makefile_path):
    with open(makefile_path, 'r') as f:
        lines = f.readlines()

    new_lines = []

    for line in lines:
        # 寻找以 _mix: 结尾的目标行
        if line.strip().endswith("_mix:"):
            target_name = line.strip()

            # 从下一行提取原始编译参数
            original_params = extract_compile_params(lines[lines.index(line) + 1])

            # 生成新的目标和规则
            new_target_line = f"{target_name} $(wildcard {target_name[:-4]}*.cu)\n"
            new_rule_line = f"\t$(foreach file,$^,nvcc -o $(patsubst %.cu,%_mix,$(file)) {original_params} $(file);)\n"

            new_lines.append(new_target_line)
            new_lines.append(new_rule_line)

    with open(makefile_path + "_new", 'w') as f:
        f.writelines(new_lines)

# 使用示例
makefile_path = './Makefile'
generate_makefile_rules(makefile_path)
