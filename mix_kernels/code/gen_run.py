from common_code import common_header, time_event_create_code, main_func_begin_code, main_func_end_code
from cp_code import get_cp_header_code, get_cp_code_before_mix_kernel, get_cp_code_after_mix_kernel, cp_gptb_params_list
from cutcp_code import get_cutcp_header_code, get_cutcp_code_before_mix_kernel, get_cutcp_code_after_mix_kernel, cutcp_gptb_params_list
from fft_code import get_fft_header_code, get_fft_code_before_mix_kernel, get_fft_code_after_mix_kernel, fft_gptb_params_list
from lbm_code import get_lbm_header_code, get_lbm_code_before_mix_kernel, get_lbm_code_after_mix_kernel, lbm_gptb_params_list
from mrif_code import get_mrif_header_code, get_mrif_code_before_mix_kernel, get_mrif_code_after_mix_kernel, mrif_gptb_params_list
from mriq_code import get_mriq_header_code, get_mriq_code_before_mix_kernel, get_mriq_code_after_mix_kernel, mriq_gptb_params_list
from sgemm_code import get_sgemm_header_code, get_sgemm_code_before_mix_kernel, get_sgemm_code_after_mix_kernel, sgemm_gptb_params_list

from data import get_kernel_info, fuse_kernel_info
from util import extract_kernel_signature, process_parameter_list

kernel_list = ['cp', 'cutcp', 'fft', 'lbm', 'mrif', 'mriq', 'sgemm']

from gen_mix import gen_pair_code_iter
from gen_test import gen_fused_code


SM_NUM = 68
def gen_code_file(kernel1, kernel2):
    kernel_file_dir = "../kernel/"
    run_file_dir = "../run/"
    # 两两生成mix_kernel
    mix_kernel_code_iter = gen_pair_code_iter(kernel1=kernel1, kernel2=kernel2)
    # 迭代
    for mix_kernel_code, ratio1, ratio2, blks_per_sm in mix_kernel_code_iter:
        print(f"gen {kernel1}_{kernel2}_{ratio1}_{ratio2}.cu, blks_per_sm: {blks_per_sm}")
        with open(kernel_file_dir + f"{kernel1}_{kernel2}_{ratio1}_{ratio2}.cu", "w") as f:
            f.write(mix_kernel_code)
        
        run_code = gen_fused_code(kernel1, kernel2, SM_NUM * blks_per_sm, get_kernel_info(kernel1)["blocksize"] * ratio1 + get_kernel_info(kernel2)["blocksize"] * ratio2, (ratio1, ratio2))

        with open(run_file_dir + f"{kernel1}_{kernel2}_{ratio1}_{ratio2}.cu", "w") as f: 
            f.write(run_code)

        # 利用makefile编译，makefile在../run/下，命令为`make {kernel1}_{kernel2}_mix`
        cmd = f"make {kernel1}_{kernel2}_mix"
        # 使用子进程执行命令
        import subprocess
        output = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT, timeout=20, cwd=run_file_dir)
        if "Error" in output or "errors" in output:
                    print(f"Error running {executable_name}, Error: ---\n{output}\n---\n", file=sys.stderr)
        print(output)
        input("Press Enter to continue...")


        

    
        # test_fused_code = gen_fused_code(kernel1, kernel2, SM_NUM * ratio1, )
        # candidates = fuse_kernel_info(kernel1, kernel2)
        # for candidate in candidates:
        #     with open(file_dir + f"{kernel1}_{kernel2}_{candidate[1]}_{candidate[2]}.cu", "w") as f:
        #         f.write(gen_fused_code(kernel1, kernel2, SM_NUM * candidate[0], candidate[1] * get_kernel_info(kernel1)["blocksize"] + candidate[2] * get_kernel_info(kernel2)["blocksize"], (candidate[1], candidate[2])))
        # candidates_ = fuse_kernel_info(kernel2, kernel1)
        # for candidate in candidates_:
        #     if (candidate[0], candidate[2], candidate[1]) in candidates:
        #         continue
        #     with open(file_dir + f"{kernel1}_{kernel2}_{candidate[2]}_{candidate[1]}.cu", "w") as f:
        #         f.write(gen_fused_code(kernel1, kernel2, SM_NUM * candidate[0], candidate[2] * get_kernel_info(kernel1)["blocksize"] + candidate[1] * get_kernel_info(kernel2)["blocksize"], (candidate[2], candidate[1])))
if __name__ == "__main__":
    kernel_pairs = []
    for i, kernel1 in enumerate(kernel_list):
        for j, kernel2 in enumerate(kernel_list):
            if i < j:
                kernel_pairs.append([kernel1, kernel2])
    # 两两生成mix_kernel
    for kernel1, kernel2 in kernel_pairs:
        gen_code_file(kernel1, kernel2)