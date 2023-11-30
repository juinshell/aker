cp_cutcp:
	nvcc -o cp_cutcp_mix -I../ptb_kernels -I../ptb_kernels/header -arch=sm_75 -lcurand -lm -lpthread -Xptxas -dlcm=ca --ptxas-options=-v cp_cutcp.cu #-maxrregcount 85
cutcp:
	nvcc -o cutcp_mix -I../ptb_kernels -arch=sm_75 -lcurand --ptxas-options=-v -Xptxas -dlcm=ca ../0_mybench/file_t/readatom.o mix_cutcp.cu #-maxrregcount 85
fft:
	nvcc -o fft_mix -I../ptb_kernels -arch=sm_75 -lcurand -lm -lpthread --ptxas-options=-v mix_fft.cu

all: cp cutcp fft
	@echo "All finished"

clean:
	rm *_mix