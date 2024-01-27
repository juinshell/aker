#!/bin/bash
nvcc -I/usr/local/cuda/include -Xcompiler -fPIC -shared -o libmylib.so mylib.cu -ldl -L/usr/local/cuda/lib64 -lcudart -arch=sm_75 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca