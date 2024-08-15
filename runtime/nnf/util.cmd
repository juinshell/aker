docker run --name hato -d -it -v ~/workspace/:/workspace/ --gpus all --privileged=true docker.io/pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel
# nnfusion
docker run --name nnfusion -d -it -v ~/workspace/:/workspace/ --gpus 1 --privileged=true docker.io/nnfusion/cuda:10.2-cudnn7-devel-ubuntu18.04
## tf1.14 for nnf example gen pb
docker run --name tf1.14 -d -it -v ~/workspace/:/workspace/ --gpus all --privileged=true tensorflow/tensorflow:1.14.0-gpu-py3