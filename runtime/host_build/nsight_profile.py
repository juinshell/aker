import json
import re
import subprocess
from time import sleep

# model_list = ["resnet50", "inception3", "bert", "vgg11", "vgg16", "vit"]
model_list = ["resnet50"]
# fft-cp	fft-cutcp	fft-lbm	fft-mriq	fft-sgemm	fft-stencil	mrif-lbm	mriq-lbm	mrif-sgemm	mrif-stencil
kernel_pair = [("fft", "cp"), ("fft", "cutcp"), ("fft", "lbm"), ("fft", "mriq"), ("fft", "sgemm"), ("fft", "stencil"), ("mrif", "lbm"), ("mriq", "lbm"), ("mrif", "sgemm"), ("mrif", "stencil"), ("hot3d", "lava"), ("nn", "path")]
# hot3d_lava	hot3d_nn	hot3d_path	lava_nn	lava_path	nn_path
def replace_json(kernel1, kernel, model="resnet50"):
    filename = f'../runtime/kinfo-{model}.json'
    with open(filename, 'r') as f:
        content = f.read()
        a = json.loads(content)
    a["nsight_compute"]["mix_kernel_name"] = kernel1[0] > kernel2[0] and f"{kernel2}_{kernel1}" or f"{kernel1}_{kernel2}"
    b = json.dumps(a, indent=4)
    with open(filename, "w") as f:
        f.write(b)



def run(kernel1, kernel2, model="resnet50"):
    command = f"ncu --set full ./tacker -s aker -m {model}"
    output = subprocess.check_output(command, shell=True, text=True, stderr=subprocess.STDOUT, timeout=30)
    mix_kernel_name = kernel1[0] > kernel2[0] and f"{kernel2}_{kernel1}" or f"{kernel1}_{kernel2}"
    with open(f"{mix_kernel_name}_nsight.txt", "w") as f:
        f.write(output)
    

if __name__ == "__main__":
    for kernel1, kernel2 in kernel_pair:
        replace_json(kernel1, kernel2)
        run(kernel1, kernel2)
        sleep(0.1)