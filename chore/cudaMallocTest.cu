#include <stdio.h>
#include <cuda_runtime.h>
#include <vector>
using namespace std;

void checkCudaErrors(cudaError_t err) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main() {
    cudaEvent_t start, stop;
    float milliseconds = 0.0;

    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    // Define the sizes to allocate in MB
    size_t sizesMB[] = {100};
    int numSizes = sizeof(sizesMB) / sizeof(sizesMB[0]);
    size_t size;

    vector<void*> ptrs(20);

    for (int i = 0; i < numSizes; i++) {
        size = sizesMB[i] * 1024 * 1024; // Convert MB to bytes

        int max_times = min(20, int(10 * 1024 / sizesMB[i])); // 10 GB of memory
        printf("Allocating %zu MB of memory %d times\n", sizesMB[i], max_times);

        // for (int times = 0; times < max_times; times++) {
        //     checkCudaErrors(cudaMalloc(&ptrs[times], size));
        //     checkCudaErrors(cudaFree(ptrs[times]));
        // }
        // cudaDeviceSynchronize();

        for (int times = 0; times < max_times; times++) {
            checkCudaErrors(cudaEventRecord(start));
            checkCudaErrors(cudaSetDevice(1));
            checkCudaErrors(cudaSetDevice(0));
            checkCudaErrors(cudaEventRecord(stop));
            checkCudaErrors(cudaEventSynchronize(stop));
            checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));
            printf("Time to allocate %zu MB of memory: %f ms\n", sizesMB[i], milliseconds);
        }
        // cudaDeviceSynchronize();
        // printf("Time to allocate %zu MB of memory: %f ms\n", sizesMB[i], milliseconds / max_times);

        for (int times = 0; times < max_times; times++) {
            checkCudaErrors(cudaFree(ptrs[times]));
        }
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
