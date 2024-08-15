bfs:
	nvcc -o bfs_mix -I../0_mybench -arch=sm_70 -lcurand -lm -lpthread --ptxas-options=-v ../0_mybench/file_t/spmv_t/convert_dataset.o ../0_mybench/file_t/spmv_t/mmio.o mix_bfs.cu
cp:
	nvcc -o cp_mix -I../0_mybench -arch=sm_70 -lcurand -lm -lpthread --ptxas-options=-v mix_cp.cu #-maxrregcount 85
cutcp:
	nvcc -o cutcp_mix -I../0_mybench -arch=sm_70 -lcurand --ptxas-options=-v -Xptxas -dlcm=ca ../0_mybench/file_t/readatom.o mix_cutcp.cu #-maxrregcount 85
fft:
	nvcc -o fft_mix -I../0_mybench -arch=sm_70 -lcurand -lm -lpthread --ptxas-options=-v mix_fft.cu
histo:
	nvcc -o histo_mix -I../0_mybench -arch=sm_70 -lcurand --ptxas-options=-v -Xptxas -dlcm=ca mix_histo.cu
img:
	nvcc -o img_mix -I../0_mybench -arch=sm_70 -lcurand --ptxas-options=-v -Xptxas -dlcm=ca mix_img.cu
lbm:
	nvcc -o lbm_mix -I../0_mybench -arch=sm_70 -lcurand --ptxas-options=-v -Xptxas -dlcm=ca ../0_mybench/file_t/spmv_t/convert_dataset.o ../0_mybench/file_t/spmv_t/mmio.o mix_lbm.cu
mrif:
	nvcc -o mrif_mix -I../0_mybench -arch=sm_70 -lcurand -lm -lpthread --ptxas-options=-v mix_mrif.cu
mriq:
	nvcc -o mriq_mix -I../0_mybench -arch=sm_70 -lcurand --ptxas-options=-v -Xptxas -dlcm=ca mix_mriq.cu
pns:
	nvcc -o pns_mix -I../0_mybench -arch=sm_70 -lcurand -lm -lpthread --ptxas-options=-v mix_pns.cu
regtil:
	nvcc -o regtil_mix -I../0_mybench -arch=sm_70 -lcurand --ptxas-options=-v -Xptxas -dlcm=ca mix_regtil.cu
sgemm:
	nvcc -o sgemm_mix -I../0_mybench -arch=sm_70 -lcurand --ptxas-options=-v -Xptxas -dlcm=ca mix_sgemm.cu
spmv:
	nvcc -o spmv_mix -I../0_mybench -arch=sm_70 -lcurand --ptxas-options=-v -Xptxas -dlcm=ca ../0_mybench/file_t/spmv_t/convert_dataset.o ../0_mybench/file_t/spmv_t/mmio.o mix_spmv.cu
stencil:
	nvcc -o stencil_mix -I../0_mybench -arch=sm_70 -lcurand --ptxas-options=-v -Xptxas -dlcm=ca ../0_mybench/file_t/spmv_t/convert_dataset.o ../0_mybench/file_t/spmv_t/mmio.o mix_stencil.cu
tpacf:
	nvcc -o tpacf_mix -I../0_mybench -arch=sm_70 -lcurand --ptxas-options=-v -Xptxas -dlcm=ca mix_tpacf.cu

all: bfs cp cutcp fft histo img lbm mrif mriq pns regtil sgemm spmv stencil tpacf
	@echo "All finished"

clean:
	rm *_mix