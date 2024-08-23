cp:
	nvcc -o cp_out -G -gencode -arch=sm_89,code=sm_89 -lm -lpthread --ptxas-options=-v my_cp.cu
tc:
	nvcc -o tcgemm_out -arch=sm_89 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca my_tcgemm.cu
tz:
	nvcc -o tzgemm_out -arch=sm_89 -lcublas -lcurand --ptxas-options=-v my_tzgemm.cu
	nvcc -o tzgemm2_out -arch=sm_89 -lcublas -lcurand --ptxas-options=-v ori_tzgemm.cu
gemm:
	nvcc -o gemm_out -arch=sm_89 -lcublas -lcurand --ptxas-options=-v my_gemm.cu
bfs:
	nvcc -o bfs_out -arch=sm_89 -lcublas -lcurand -lm -lpthread --ptxas-options=-v my_bfs.cu
cp:
	nvcc -o cp_out -arch=sm_89 -lm -lpthread --ptxas-options=-v my_cp.cu
cutcp:
	nvcc -o cutcp_out -arch=sm_89 -lcublas -lcurand -lm --ptxas-options=-v -Xptxas -dlcm=ca file_t/readatom.o my_cutcp.cu
fft:
	nvcc -o fft_out -arch=sm_89 -lcublas -lcurand -lm -lpthread --ptxas-options=-v my_fft.cu
histo:
	nvcc -o histo_out -arch=sm_89 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca my_histo.cu
img:
	nvcc -o img_out -arch=sm_89 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca my_img.cu
lbm:
	nvcc -o lbm_out -arch=sm_89 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca my_lbm.cu
mrif:
	nvcc -o mrif_out -arch=sm_89 -lm -lpthread --ptxas-options=-v my_mrif.cu
mriq:
	nvcc -o mriq_out -arch=sm_89 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca my_mriq.cu
pns:
	nvcc -o pns_out -arch=sm_89 -lcublas -lcurand -lm -lpthread --ptxas-options=-v my_pns.cu
regtil:
	nvcc -o regtil_out -arch=sm_89 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca my_regtil.cu
sad:
	nvcc -o sad_out -arch=sm_89 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca my_sad.cu
sgemm:
	nvcc -o sgemm_out -arch=sm_89 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca my_sgemm.cu
spmv:
	nvcc -o spmv_out -arch=sm_89 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca file_t/spmv_t/convert_dataset.o file_t/spmv_t/mmio.o my_spmv.cu
stencil:
	nvcc -o stencil_out -arch=sm_89 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca my_stencil.cu
tpacf:
	nvcc -o tpacf_out -arch=sm_89 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca my_tpacf.cu

all:  cp cutcp fft lbm mrif mriq sgemm stencil
	@echo "All finished"

clean:
	rm *_out