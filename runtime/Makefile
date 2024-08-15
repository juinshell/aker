ori_fft:
	nvcc -cubin -o ./cubins/ori_fft.cubin -I../mix_kernels -arch=sm_75 -lcublas -lcurand -lm -lpthread --ptxas-options=-v ./ori_kernel/ori_fft.cu
ori_cp:
	nvcc -cubin -o ./cubins/ori_cp.cubin -I../mix_kernels -arch=sm_75 -lm -lpthread --ptxas-options=-v ./ori_kernel/ori_cp.cu
ori_cutcp:
	nvcc -cubin -o ./cubins/ori_cutcp.cubin -I../mix_kernels -arch=sm_75 -lcublas -lcurand -lm --ptxas-options=-v -Xptxas -dlcm=ca ./ori_kernel/ori_cutcp.cu
ori: ori_cp ori_fft ori_cutcp
	@echo "ori done"

cp_fft: 
	nvcc -cubin -o ./cubins/cp_fft.cubin -I../mix_kernels --ptxas-options=-v -lcublas -lcurand -lm -arch=sm_75 -lpthread ./mix_kernel/cp_fft_3_1.cu

cp_sgemm:
	nvcc -cubin -o ./cubins/cp_sgemm.cubin -I../mix_kernels --ptxas-options=-v -lcublas -lcurand -lm -arch=sm_75 -lpthread -Xptxas -dlcm=ca ./mix_kernel/cp_sgemm_1_1.cu

cutcp_fft: 
	nvcc -cubin -o ./cubins/cutcp_fft.cubin -I../mix_kernels --ptxas-options=-v -lcublas -lm -lpthread -arch=sm_75 -lcurand -Xptxas -dlcm=ca ./mix_kernel/cutcp_fft_1_1.cu
	
cutcp_sgemm: 
	nvcc -cubin -o ./cubins/cutcp_sgemm.cubin -I../mix_kernels --ptxas-options=-v -lcublas -lm -arch=sm_75 -lcurand -Xptxas -dlcm=ca ./mix_kernel/cutcp_sgemm_1_1.cu

fft_lbm: 
	nvcc -cubin -o ./cubins/fft_lbm.cubin -I../mix_kernels --ptxas-options=-v -lcublas -lcurand -lm -arch=sm_75 -lpthread -Xptxas -dlcm=ca ./mix_kernel/fft_lbm_6_1.cu

fft_mriq: 
	nvcc -cubin -o ./cubins/fft_mriq.cubin -I../mix_kernels --ptxas-options=-v -lcublas -lcurand -lm -arch=sm_75 -lpthread -Xptxas -dlcm=ca ./mix_kernel/fft_mriq_3_2.cu

fft_sgemm: 
	nvcc -cubin -o ./cubins/fft_sgemm.cubin -I../mix_kernels --ptxas-options=-v -lcublas -lcurand -lm -arch=sm_75 -lpthread -Xptxas -dlcm=ca ./mix_kernel/fft_sgemm_1_4.cu

lbm_mrif: 
	nvcc -cubin -o ./cubins/lbm_mrif.cubin -I../mix_kernels --ptxas-options=-v -lcublas -lm -lpthread -arch=sm_75 -lcurand -Xptxas -dlcm=ca ./mix_kernel/lbm_mrif_1_3.cu

lbm_mriq: 
	nvcc -cubin -o ./cubins/lbm_mriq.cubin -I../mix_kernels --ptxas-options=-v -lcublas -arch=sm_75 -lcurand -Xptxas -dlcm=ca ./mix_kernel/lbm_mriq_1_2.cu

lbm_sgemm: 
	nvcc -cubin -o ./cubins/lbm_sgemm.cubin -I../mix_kernels --ptxas-options=-v -lcublas -arch=sm_75 -lcurand -Xptxas -dlcm=ca ./mix_kernel/lbm_sgemm_1_7.cu

mrif_sgemm: 
	nvcc -cubin -o ./cubins/mrif_sgemm.cubin -I../mix_kernels --ptxas-options=-v -lcublas -lcurand -lm -arch=sm_75 -lpthread -Xptxas -dlcm=ca ./mix_kernel/mrif_sgemm_1_4.cu

mriq_sgemm: 
	nvcc -cubin -o ./cubins/mriq_sgemm.cubin -I../mix_kernels --ptxas-options=-v -lcublas -arch=sm_75 -lcurand -Xptxas -dlcm=ca ./mix_kernel/mriq_sgemm_1_2.cu

mix: cp_sgemm cp_fft cutcp_fft cutcp_sgemm fft_lbm fft_mriq fft_sgemm lbm_mrif lbm_mriq lbm_sgemm mrif_sgemm mriq_sgemm
	@echo "mix done"