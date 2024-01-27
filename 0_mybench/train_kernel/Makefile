all: 
	# 

test:
	nvcc -o test_out -arch=sm_75 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca test2.cu
tc:
	nvcc -o tcgemm_out -arch=sm_75 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca my_tcgemm.cu
tx:
	nvcc -o txgemm_out -arch=sm_75 -lcublas -lcurand --ptxas-options=-v my_txgemm.cu
tz:
	nvcc -o tzgemm_out -arch=sm_75 -lcublas -lcurand --ptxas-options=-v my_tzgemm.cu
	nvcc -o tzgemm_ori -arch=sm_75 -lcublas -lcurand --ptxas-options=-v ori_tzgemm.cu
gemm:
	nvcc -o gemm_out -arch=sm_75 -lcublas -lcurand --ptxas-options=-v my_gemm.cu
bfs:
	nvcc -o bfs_out -arch=sm_75 -lcublas -lcurand -lm -lpthread --ptxas-options=-v my_bfs.cu
cp:
	nvcc -o cp_out -arch=sm_75 -lm -lpthread --ptxas-options=-v my_cp.cu
cutcp:
	nvcc -o cutcp_out -arch=sm_75 -lcublas -lcurand -lm --ptxas-options=-v -Xptxas -dlcm=ca file_t/readatom.o my_cutcp.cu
fft:
	nvcc -o fft_out -arch=sm_75 -lcublas -lcurand -lm -lpthread --ptxas-options=-v my_fft.cu
histo:
	nvcc -o histo_out -arch=sm_75 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca my_histo.cu
img:
	nvcc -o img_out -arch=sm_75 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca my_img.cu
lbm:
	nvcc -o lbm_out -arch=sm_75 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca my_lbm.cu
mrif:
	nvcc -o mrif_out -arch=sm_75 -lm -lpthread --ptxas-options=-v my_mrif.cu
mriq:
	nvcc -o mriq_out -arch=sm_75 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca my_mriq.cu
pns:
	nvcc -o pns_out -arch=sm_75 -lcublas -lcurand -lm -lpthread --ptxas-options=-v my_pns.cu
regtil:
	nvcc -o regtil_out -arch=sm_75 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca my_regtil.cu
sad:
	nvcc -o sad_out -arch=sm_75 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca my_sad.cu
sgemm:
	nvcc -o sgemm_out -arch=sm_75 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca my_sgemm.cu
spmv:
	nvcc -o spmv_out -arch=sm_75 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca file_t/spmv_t/convert_dataset.o file_t/spmv_t/mmio.o my_spmv.cu
stencil:
	nvcc -o stencil_out -arch=sm_75 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca my_stencil.cu
tpacf:
	nvcc -o tpacf_out -arch=sm_75 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca my_tpacf.cu
mrig:
	nvcc -o mrig_out mrig_t/mrig_sort.cu mrig_t/mrig_scanarray.cu mrig_t/mrig_cpu_kernels.c my_mrig.cu
im2col:
	nvcc -o  im2col_out -I../ -arch=sm_75 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca my_im2col.cu
	nvcc -o im2col_test_out -I../ -arch=sm_75 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca my_im2col_test.cu
relu:
	nvcc -o relu_out -arch=sm_75 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca my_relu.cu
	nvcc -o relu_mix -arch=sm_75 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca mix_relu.cu
scale:
	nvcc -o scale_out -arch=sm_75 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca my_scale.cu
	nvcc -o scale_mix -arch=sm_75 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca mix_scale.cu
pool:
	nvcc -o pool_out -arch=sm_75 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca my_maxpool.cu
	nvcc -o pool_mix -arch=sm_75 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca mix_maxpool.cu
elt:
	nvcc -o elt_out -arch=sm_75 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca my_eltwise.cu
	nvcc -o elt_mix -arch=sm_75 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca mix_eltwise.cu
ccat:
	nvcc -o ccat_out -arch=sm_75 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca my_concat.cu
	nvcc -o ccat_mix -arch=sm_75 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca mix_concat.cu
bn:
	nvcc -o bn_out -arch=sm_75 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca my_batchnorm.cu
	nvcc -o bn_mix -arch=sm_75 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca mix_batchnorm.cu

clean:
	rm *_out
	rm mrig_t/*.o