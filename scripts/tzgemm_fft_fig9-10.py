import re
import subprocess
import sys
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

f = open('../runtime/kinfo-common.json', 'r')
content = f.read()
a = json.loads(content)
f.close()

command = "../runtime/build/tacker -s tacker -m resnet50"

def compile(mix_fft_blk_num, mix_tzgemm_blk_num):
    with open('../runtime/kinfo-common.json', 'w') as f:
        a["ratio_test"]["mix_fft_blk_num"] = mix_fft_blk_num
        a["ratio_test"]["mix_tzgemm_blk_num"] = mix_tzgemm_blk_num
        f.write(json.dumps(a, indent=4))


def run():
    # 运行
    try:
        output = subprocess.check_output(command, shell=True, text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print(f"Error running {command}, Error: ---\n{e.output}\n---\n, Exit!", file=sys.stderr)
        exit(1)
    # if extract_data(output)['load_ratio'] > 1.0:
    #     return 
    data_dic_list = []
    for i in range(10):
        try:
            output = subprocess.check_output(command, shell=True, text=True, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            print(f"Error running {command}, Error: ---\n{e.output}\n---\n, Exit!", file=sys.stderr)
            exit(1)
        data_dic_list.append(extract_data(output))
    
    # 对mix_duration进行排序，取中间10次的平均值
    data_dic_list.sort(key=lambda x: x['mix_duration'])
    data_dic_list = data_dic_list[2:7]
    load_ratio = sum([x['load_ratio'] for x in data_dic_list]) / len(data_dic_list)
    mix_duration = sum([x['mix_duration'] for x in data_dic_list]) / len(data_dic_list)
    sgemm_gptb_time = sum([x['sgemm_gptb_time'] for x in data_dic_list]) / len(data_dic_list)
    fft_gptb_time = sum([x['fft_gptb_time'] for x in data_dic_list]) / len(data_dic_list)
    sgemm_blk_num = sum([x['sgemm_blk_num'] for x in data_dic_list]) / len(data_dic_list)
    fft_blk_num = sum([x['fft_blk_num'] for x in data_dic_list]) / len(data_dic_list)
    return {
        'load_ratio': load_ratio,
        'mix_duration': mix_duration,
        'sgemm_gptb_time': sgemm_gptb_time,
        'fft_gptb_time': fft_gptb_time,
        'sgemm_blk_num': sgemm_blk_num,
        'fft_blk_num': fft_blk_num
    }

import re
import pandas as pd

data = []

def extract_data(data_content):
    data_ = []
    pattern = r'\s+load_ratio:\s+(\d+\.\d+)\s+mix_duration:\s+(\d+\.\d+)\s+.*sgemm gptb time:\s+(\d+\.\d+), fft gptb time:\s+(\d+\.\d+), sgemm_blk_num:\s+(\d+),\s+fft_blk_num:\s+(\d+)'

    matches = re.findall(pattern, data_content, re.MULTILINE)

    assert len(matches) == 1

    for match in matches:
        data_.append({
            'load_ratio': float(match[0]),
            'mix_duration': float(match[1]),
            'sgemm_gptb_time': float(match[2]),
            'fft_gptb_time': float(match[3]),
            'sgemm_blk_num': int(match[4]),
            'fft_blk_num': int(match[5])
        })
    return data_[0]

def write_to_excel(output_file):
    df = pd.DataFrame(data)
    df.to_excel(output_file, index=False)

if __name__ == "__main__":
    max_fft_blks = 10240 * 20
    max_tzgemm_blks = 160000
    fft_blks = 0
    tzgemm_blks = 0
    SM_NUM = 142
    fig = input("choose to gen fig9 or fig10")
    if fig == "9":
        # fig9
        tzgemm_blks = 51200 * 2
        for i in range(SM_NUM * 50, max_fft_blks + 1):
            if i % (SM_NUM * 30) == 0:
                print(f"--- fft_blks: {i}, tzgemm_blks: {tzgemm_blks} ---")
                compile(i, tzgemm_blks)
                output = run()
                if output['load_ratio'] > 1.2:
                    break
                print(output)
                data.append(output)
        
        output_file = '[Aker]fig9-fft-tzgemm.xlsx'
        write_to_excel(output_file)
    elif fig == "10":
        # fig10
        scaling_rate = 2
        load_ratios = [0.35, 0.72, 1.06, 1.41]
        for load_ratio in load_ratios:
            points_num = 0
            for i in range(1, max_tzgemm_blks + 1):
                if i % (SM_NUM * 50) == 0:
                    if points_num > 17:
                        break
                    fft_blks = int(i * load_ratio * scaling_rate)
                    tzgemm_blks = i
                    print(f"--- fft_blks: {fft_blks}, tzgemm_blks: {tzgemm_blks} ---")
                    compile(fft_blks, tzgemm_blks)
                    output = run()
                    output['load_ratio'] = load_ratio
                    print(output)
                    data.append(output)
                    points_num += 1
        
        output_file = '[Aker]fig10-a.xlsx'
        write_to_excel(output_file)




