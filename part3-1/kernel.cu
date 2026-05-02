#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdio.h> 

using std::cout;
using std::generate;
using std::vector;

__global__ void sumReductionNaive(float* input, float* output, int n) {
	// Shared memory for block-level reduction
    __shared__ float sharedData[256]; // Assuming block size <= 256

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data from global to shared memory
    if (idx < n) {
        sharedData[tid] = input[idx];
    }
    else {
        sharedData[tid] = 0.0f;
    }

    __syncthreads(); // Ensure all data is loaded

    // Naive reduction within block
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0) {
            sharedData[tid] += sharedData[tid + stride];
        }
		__syncthreads(); // Synchronize after each step
    }

    // Write block result to output
    if (tid == 0) {
        atomicAdd(output, sharedData[0]);
    }
}

__global__ void activationFunction(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
	}
}

int main() {
    int N = 8192;
    size_t bytes = N * sizeof(float);

    vector<float> h_a(N);
    generate(h_a.begin(), h_a.end(), []() { return static_cast<float>((rand() % 100 - 50) / 100.0f); });

    float h_final_result = 0.0f;
    float *d_a, *d_r, *d_r2;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_r, sizeof(float));
	cudaMalloc(&d_r2, sizeof(float));

    cudaMemset(d_r, 0, sizeof(float));

    cudaEvent_t start, stop, Host2dev, KernelExec;
    cudaEventCreate(&start);
    cudaEventCreate(&Host2dev);
    cudaEventCreate(&KernelExec);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    // Copy to device
    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(Host2dev, 0);
    
    int THREADS = 32;
    int BLOCKS = (N + THREADS - 1) / THREADS;

    cout << "Running sumReductionNaive with " << BLOCKS << " blocks and " << THREADS << " threads...\n";

    // Launch 1D grid and 1D block
    sumReductionNaive << <BLOCKS, THREADS >> > (d_a, d_r, N);
	activationFunction << <1, 1 >> > (d_r, d_r2, 1);

    cudaEventRecord(KernelExec, 0);

	cudaMemcpy(&h_final_result, d_r2, sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float Total_gpu_time, Host2Dev_time, Kernel_time, Dev2Host_time;
    cudaEventElapsedTime(&Total_gpu_time, start, stop);
    cudaEventElapsedTime(&Host2Dev_time, start, Host2dev);
    cudaEventElapsedTime(&Kernel_time, Host2dev, KernelExec);
    cudaEventElapsedTime(&Dev2Host_time, KernelExec, stop);

    printf("Time elapsed Host To Device: %f ms\n", Host2Dev_time);
    printf("Time elapsed Kernel Execution: %f ms\n", Kernel_time);
    printf("Time elapsed Device To Host: %f ms\n", Dev2Host_time);
    printf("Total Time: %f ms\n\n", Total_gpu_time);

	float expectedSum = 0.0f;
    for (const auto& val : h_a) {
        expectedSum += val;
	}
	expectedSum = 1.0f / (1.0f + expf(-expectedSum));

	printf("Expected Sum Result: %f\n", expectedSum);
    printf("Final Sum Result: %f\n", h_final_result);

    cudaFree(d_a);
    cudaFree(d_r);

    return 0;
}