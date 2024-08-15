
im2col:
	nvcc -o im2col_mix -I../../0_mybench -arch=sm_75 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca mix_im2col.cu
relu:
	nvcc -o relu_mix -I../../0_mybench -arch=sm_75 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca mix_relu.cu
scale:
	nvcc -o scale_mix -I../../0_mybench -arch=sm_75 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca mix_scale.cu
pool:
	nvcc -o pool_mix -I../../0_mybench -arch=sm_75 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca mix_maxpool.cu
elt:
	nvcc -o elt_mix -I../../0_mybench -arch=sm_75 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca mix_eltwise.cu
ccat:
	nvcc -o ccat_mix -I../../0_mybench -arch=sm_75 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca mix_concat.cu
bn:
	nvcc -o bn_mix -I../../0_mybench -arch=sm_75 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca mix_batchnorm.cu

clean:
	rm *_mix