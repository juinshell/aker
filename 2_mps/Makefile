.PHONY: all
all: sgemm tz cutcp fft lbm mrif regtiling stencil tpacf cp mriq init_ipc

init_ipc:
	g++ -o init_ipc -I../0_mybench init_ipc.cc -lrt -lpthread

sgemm:
	nvcc -o sgemm_out -I../0_mybench -arch=sm_75 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca my_sgemm.cu
tz:
	nvcc -o tzgemm_out -I../0_mybench -arch=sm_75 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca my_tzgemm.cu

cutcp:
	nvcc -o cutcp_out -I../0_mybench -arch=sm_75 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca my_cutcp.cu
fft:
	nvcc -o fft_out -I../0_mybench -arch=sm_75 -lcublas -lcurand -lm -lpthread --ptxas-options=-v my_fft.cu
lbm:
	nvcc -o lbm_out -I../0_mybench -arch=sm_75 -lcublas -lcurand -lm -lpthread --ptxas-options=-v my_lbm.cu
mrif:
	nvcc -o mrif_out -I../0_mybench -arch=sm_75 -lcurand -lm -lpthread --ptxas-options=-v my_mrif.cu
regtiling:
	nvcc -o regtiling_out -I../0_mybench -arch=sm_75 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca my_regtiling.cu
stencil:
	nvcc -o stencil_out -I../0_mybench -arch=sm_75 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca my_stencil.cu
tpacf:
	nvcc -o tpacf_out -I../0_mybench -arch=sm_75 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca my_tpacf.cu
cp:
	nvcc -o cp_out -I../0_mybench -arch=sm_75 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca my_cp.cu
mriq:
	nvcc -o mriq_out -I../0_mybench -arch=sm_75 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca my_mriq.cu

.PHONY: clean
clean:
	rm -f sgemm_out
	rm -f tzgemm_out
	rm -f cutcp_out
	rm -f fft_out
	rm -f lbm_out
	rm -f mrif_out
	rm -f regtiling_out
	rm -f stencil_out
	rm -f tpacf_out
	rm -rf cp_out
	rm -rf mriq_out