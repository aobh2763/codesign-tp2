#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdio.h>

using std::cout;
using std::generate;
using std::vector;

#define WARP_SIZE 32

template <unsigned int blockSize>
__device__ void warpReduce(volatile float* sdata, unsigned int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8)  sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4)  sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2)  sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__global__ void reducebase7(float* g_idata, float* g_odata, unsigned int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + tid;
    unsigned int gridSize = blockSize * 2 * gridDim.x;

    float mySum = 0;

    // Grid-stride loop to process multiple elements and handle bounds
    while (i < n) {
        mySum += g_idata[i];
        if (i + blockSize < n) {
            mySum += g_idata[i + blockSize];
        }
        i += gridSize;
    }
    sdata[tid] = mySum;
    __syncthreads();

    if (blockSize >= 512) { if (tid < 256) sdata[tid] += sdata[tid + 256]; __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) sdata[tid] += sdata[tid + 128]; __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64)  sdata[tid] += sdata[tid + 64];  __syncthreads(); }

    if (tid < WARP_SIZE) warpReduce<blockSize>(sdata, tid);
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

int main() {
    int N = 8192 * 256;
    const size_t blockSize = 256;
    size_t bytes = N * sizeof(float);

    vector<float> h_in(N);
    float h_out_final = 0;

    cout << "Step1: h_in generation \n";
    generate(h_in.begin(), h_in.end(), []() { return (float)(rand() % 10); });

    cout << "Step2: Mem Allocation on device \n";
    float* d_in, * d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    cout << "Step3: Launch Event to measure Time \n";
    float Total_gpu_time, Host2Dev_time, Kernel_time, Dev2Host_time;
    cudaEvent_t start, stop, Host2dev, KernelExec;
    cudaEventCreate(&start);
    cudaEventCreate(&Host2dev);
    cudaEventCreate(&KernelExec);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    cout << "Step4: Copy Data To Device \n";
    cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(Host2dev, 0);

    // --- FIXED RECURSIVE REDUCTION LOOP ---
    int current_n = N;
    while (current_n > 1) {
        // Correct calculation for grid size (2 elements per thread)
        int grid_size = (current_n + (blockSize * 2) - 1) / (blockSize * 2);

        // Pass the shared memory size as the 3rd argument
        size_t smemSize = blockSize * sizeof(float);

        reducebase7<blockSize> << <grid_size, blockSize, smemSize >> > (d_in, d_out, current_n);

        // Pointer Swap (Fast and safe for recursion)
        float* temp = d_in;
        d_in = d_out;
        d_out = temp;

        current_n = grid_size;
    }

    cudaEventRecord(KernelExec, 0);

    // Result is in d_in because of the final swap in the loop
    cudaMemcpy(&h_out_final, d_in, sizeof(float), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&Total_gpu_time, start, stop);
    cudaEventElapsedTime(&Host2Dev_time, start, Host2dev);
    cudaEventElapsedTime(&Kernel_time, Host2dev, KernelExec);
    cudaEventElapsedTime(&Dev2Host_time, KernelExec, stop);

    printf("Time elapsed on Host To Device Transfer: %f ms.\n", Host2Dev_time);
    printf("Time elapsed on Reduction Kernel(s): %f ms.\n", Kernel_time);
    printf("Time elapsed on Device To Host Transfer: %f ms.\n", Dev2Host_time);
    printf("Total Time: %f ms.\n\n", Total_gpu_time);

    // Using double for CPU verification to ensure absolute precision
    double cpu_sum = 0;
    for (float f : h_in) cpu_sum += (double)f;

    printf("GPU Result: %f\n", h_out_final);
    printf("CPU Result: %f\n", (float)cpu_sum);

    if (abs((float)cpu_sum - h_out_final) < 1.0f) { // Adjusted tolerance for large N
        cout << "COMPLETED SUCCESSFULLY\n";
    }
    else {
        cout << "VERIFICATION FAILED\n";
    }

    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}