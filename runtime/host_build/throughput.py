import json
import re
import subprocess
from time import sleep

# model_list = ["resnet50", "inception3", "bert", "vgg11", "vgg16", "vit"]
model_list = ["vit"]
# fft-cp	fft-cutcp	fft-lbm	fft-mriq	fft-sgemm	fft-stencil	mrif-lbm	mriq-lbm	mrif-sgemm	mrif-stencil
kernel_pair = [("fft", "cp"), ("fft", "cutcp"), ("fft", "lbm"), ("fft", "mriq"), ("fft", "sgemm"), ("fft", "stencil"), ("mrif", "lbm"), ("mriq", "lbm"), ("mrif", "sgemm"), ("mrif", "stencil")]
# hot3d_lava	hot3d_nn	hot3d_path	lava_nn	lava_path	nn_path
# kernel_pair = [("hot3d", "lava"), ("hot3d", "nn"), ("hot3d", "path"), ("lava", "nn"), ("lava", "path"), ("nn", "path")]
def replace_json(model, kernel1, kernel2):
    filename = f'../kinfo-{model}.json'
    with open(filename, 'r') as f:
        content = f.read()
        a = json.loads(content)
    a["throughput_test"]["a"] = kernel1
    a["throughput_test"]["b"] = kernel2
    b = json.dumps(a, indent=4)
    with open(filename, "w") as f:
        f.write(b)

def verify(model, kernel1, kernel2, output):
    pattern = r"\[Result\] cd1: (\w+), cd2: (\w+), dnn: (\w+)"
    matches = re.findall(pattern, output)
    if len(matches) != 1:
        print(f"Error: No matches found in output")
        exit(1)
    match = matches[0]
    if match[0] != kernel1 or match[1] != kernel2 or match[2] != model:
        print(f"Error: Kernel mismatch. Expected: {kernel1}, {kernel2}, {model}. Got: {match[0]}, {match[1]}, {match[2]}")
        input("Press Enter to continue...")

data_dict = {
    "resnet50": [],
    "inception3": [],
    "bert": [],
    "vgg11": [],
    "vgg16": [],
    "vit": []
}

aker_headroom_list = []
taker_headroom_list = []
def deal_result(model, kernel1, kernel2, output):
    pattern = r"\((\d+\.\d+)\) = (\d+\.\d+) ms to execute"
    matches = re.findall(pattern, output)
    if len(matches) != 2:
        print(f"Error: No matches found in output")
        exit(1)
    # print(matches)
    aker_headroom, aker_throughput = float(matches[0][0]), float(matches[0][1])
    tacker_headroom, tacker_throughput = float(matches[1][0]), float(matches[1][1])
    print(f"aker throughput: {aker_throughput}, tacker throughput: {tacker_throughput}")
    print(f"aker headroom: {aker_headroom}, tacker headroom: {tacker_headroom}")
    data_dict[model].append({
        "mix_name": f"{kernel1}-{kernel2}",
        "aker_throughput": aker_throughput,
        "tacker_throughput": tacker_throughput
    })
    aker_headroom_list.append(aker_headroom)
    taker_headroom_list.append(tacker_headroom)


def run(model, kernel1, kernel2):
    command = f"./tacker -s aker -m {model}| grep Result"
    output = subprocess.check_output(command, shell=True, text=True, stderr=subprocess.STDOUT, timeout=30)
    print(output)
    verify(model, kernel1, kernel2, output)
    deal_result(model, kernel1, kernel2, output)
    

if __name__ == "__main__":
    for model in model_list:
        for kernel1, kernel2 in kernel_pair:
            replace_json(model, kernel1, kernel2)
            run(model, kernel1, kernel2)
            sleep(0.1)
    for model in model_list:
        print(model, "---")
        for data in data_dict[model]:
            print(data["mix_name"], end=" ")
        print()
        print("aker throughput: ", end=" ")
        for data in data_dict[model]:
            print(data["aker_throughput"], end=" ")
        print()
        print("tacker throughput: ", end=" ")
        for data in data_dict[model]:
            print(data["tacker_throughput"], end=" ")
        print()
        print("---------")
    print("aker headroom: ", end=" ")
    for headroom in aker_headroom_list:
        print(headroom, end=" ")
    print()
    print("tacker headroom: ", end=" ")
    for headroom in taker_headroom_list:
        print(headroom, end=" ")
    print()