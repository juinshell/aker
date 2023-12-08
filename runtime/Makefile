ori_fft:
	nvcc -cubin -o ./cubins/ori_fft.cubin -I. -arch=sm_75 -lcublas -lcurand -lm -lpthread --ptxas-options=-v ./ori_kernel/ori_fft.cu
ori_cp:
	nvcc -cubin -o ./cubins/ori_cp.cubin -I. -arch=sm_75 -lm -lpthread --ptxas-options=-v ./ori_kernel/ori_cp.cu

ori: ori_cp ori_fft
	@echo "ori done"