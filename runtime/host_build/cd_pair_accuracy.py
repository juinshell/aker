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

kernel_pair = [("fft", "cp"), ("fft", "cutcp"), ("fft", "lbm"), ("mriq", "fft"), ("fft", "sgemm"), ("stencil", "fft"), ("mrif", "lbm"), ("mriq", "lbm"), ("sgemm", "mrif"), ("mrif", "stencil"),
               ("hot3d", "lava"), ("hot3d", "nn"), ("hot3d", "path"), ("lava", "nn"), ("lava", "path"), ("nn", "path")]
# kernel_pair = [("sgemm", "fft"), ("sgemm", "mrif"), ("mriq", "fft"), ("stencil", "fft")]

SM_NUM = 142
steps_dict = {
    "cp": SM_NUM * 5,
    "cutcp": SM_NUM * 4,
    "fft": SM_NUM * 10,
    "lbm": SM_NUM * 10,
    "mrif": SM_NUM,
    "mriq": SM_NUM * 2,
    "sgemm": SM_NUM,
    "stencil": SM_NUM * 3,
    "hot3d": SM_NUM * 100,
    "lava": SM_NUM * 10,
    "nn": SM_NUM * 1000,
    "path": SM_NUM * 10
}


def replace_json(kernel1, kernel2, a_blk_num, b_blk_num):
    a["cd_pair_accuracy"]["a_name"] = kernel1
    a["cd_pair_accuracy"]["b_name"] = kernel2
    a["cd_pair_accuracy"]["a_blk_num"] = a_blk_num
    a["cd_pair_accuracy"]["b_blk_num"] = b_blk_num
    b = json.dumps(a, indent=4)
    f = open('../kinfo-resnet50.json', 'w')
    f.write(b)
    f.close()

def extract_data(data_content):
    data_ = []
    pattern = r'base_blks: (\d+), duration: (\d+\.\d+)'

    matches = re.findall(pattern, data_content, re.MULTILINE)

    for match in matches:
        data_.append({
            'base_blks': int(match[0]),
            'duration': float(match[1]),
        })
        break

    return data_[0]

output_file = "cd_pair_accuracy.csv"
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
    
    base_blk = sum([x['base_blks'] for x in tmp_list]) / len(tmp_list)
    duration = sum([x['duration'] for x in tmp_list]) / len(tmp_list)
    return [int(base_blk), duration]

# 定义ctrl+c退出时的处理函数
def signal_handler(signal, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)

slient_pair = []
if __name__ == "__main__":
    # register signal handler
    data = []
    import signal
    signal.signal(signal.SIGINT, signal_handler)
    columns = []
    for kernel1, kernel2 in kernel_pair:
        if (kernel1, kernel2) in slient_pair:
            continue
        columns.append(f"{kernel1}_{kernel2}_base_blks")
        columns.append(f"{kernel1}_{kernel2}_duration")
    print("columns:", columns)
    for i in range(1, 11):
        for kernel1, kernel2 in kernel_pair:
            if (kernel1, kernel2) in slient_pair:
                continue
            max_cd1_blks = a[kernel1]["real_ori_blks"]
            max_cd2_blks = a[kernel2]["real_ori_blks"]
            b_blks = int(i * steps_dict[kernel2])
            a_blks = int(i * steps_dict[kernel1])
            replace_json(kernel1, kernel2, a_blks, b_blks)
            print(f"--- kernel1:{kernel1} kernel2:{kernel2} blks1:{a_blks} blks2:{b_blks} ---")
            data_i = run()
            data.append(data_i[0])
            data.append(data_i[1])
        print("result:", data)
        data_ = np.array(data).reshape(1, len(columns))
        append_to_csv(data_, columns)
        data.clear()