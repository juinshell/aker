test: 
	nvcc -o test -arch=sm_75 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca mix_warp.cu
clean:
	rm -f test

