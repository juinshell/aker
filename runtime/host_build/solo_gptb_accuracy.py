import json
import re
import subprocess
import sys
from time import sleep

import numpy as np
import pandas as pd
f = open('../kinfo-resnet50.json', 'r')
content = f.read()
a = json.loads(content)
f.close()

kernel_list = ["cp", "cutcp", "fft", "lbm", "mrif", "mriq", "sgemm", "stencil", "tzgemm", "hot3d", "lava", "nn", "path"]

active_kernel = []
SM_NUM = 142
steps_dict = {
    "cp": 680 / 68 * SM_NUM,
    "cutcp": 680 / 68 * SM_NUM,
    "fft": 1360 / 68 * SM_NUM,
    "lbm": 2720 / 68 * SM_NUM,
    "mrif": 68 / 68 * SM_NUM,
    "mriq": 272 / 68 * SM_NUM,
    "sgemm": 136 / 68 * SM_NUM,
    "stencil": 4 * SM_NUM,
    "tzgemm": 10 * SM_NUM,
    "hot3d": SM_NUM * 100,
    "lava": SM_NUM * 10,
    "nn": SM_NUM * 1000,
    "path": SM_NUM * 10
}


def replace_json(kernel, blk_num):
    a["solo_gptb_accuracy"]["name"] = kernel
    a["solo_gptb_accuracy"]["blk_num"] = int(blk_num)
    b = json.dumps(a, indent=4)
    f = open('../kinfo-resnet50.json', 'w')
    f.write(b)
    f.close()

def extract_data(data_content):
    data_ = []
    pattern = r'blk_num: (\d+), duration: (\d+\.\d+)'

    matches = re.findall(pattern, data_content, re.MULTILINE)

    for match in matches:
        data_.append({
            'blk_num': int(match[0]),
            'duration': float(match[1]),
        })
        break

    return data_[0]

output_file = "solo_gptb_accuracy.csv"
iter_num = 1

def write_to_excel(writer, sheet_name, data):
    df = pd.DataFrame(data)
    df.to_excel(writer, sheet_name=sheet_name)

def append_to_csv(data, columns):
    # 小数保留3位，整数不变
    df = pd.DataFrame(data, columns=columns)

    # 分隔符为\t
    df.to_csv(output_file, mode='a', header=False, index=False, sep=',', float_format='%.3f')

def run():
    command = f"./tacker -s aker -m resnet50"
    tmp_list = []
    # output = subprocess.check_output(command, shell=True, text=True, stderr=subprocess.STDOUT, timeout=30)
    # print(output)
    for i in range(iter_num):
        output = subprocess.check_output(command, shell=True, text=True, stderr=subprocess.STDOUT, timeout=30)
        tmp_list.append(extract_data(output))
    
    blk_num = sum([x['blk_num'] for x in tmp_list]) / len(tmp_list)
    duration = sum([x['duration'] for x in tmp_list]) / len(tmp_list)
    return [int(blk_num), duration]

# 定义ctrl+c退出时的处理函数
def signal_handler(signal, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)

if __name__ == "__main__":
    # register signal handler
    data = []
    import signal
    signal.signal(signal.SIGINT, signal_handler)
    columns = []
    if len(active_kernel) > 0:
        kernel_list = active_kernel
    for kernel in kernel_list:
        columns.append(f"{kernel}_gptb_blks")
        columns.append(f"{kernel}_duration")
    print("columns:", columns)
    for i in range(1 + 1, 11 + 1):
        for kernel in kernel_list:
            replace_json(kernel, i * steps_dict[kernel] * 2)
            print(f"--- kernel: {kernel}, blk_num: {i * steps_dict[kernel]}")
            data_i = run()
            data.append(data_i[0])
            data.append(data_i[1])
        print("result:", data)
        data_ = np.array(data).reshape(1, len(columns))
        append_to_csv(data_, columns)
        data.clear()