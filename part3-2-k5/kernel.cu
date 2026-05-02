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

// Kernel definition as provided
template <typename T>
__device__ void warpReduce4(volatile T* cache, unsigned int tid) {
    cache[tid] += cache[tid + 32];
    //__syncthreads();
    cache[tid] += cache[tid + 16];
    //__syncthreads();
    cache[tid] += cache[tid + 8];
    //__syncthreads();
    cache[tid] += cache[tid + 4];
    //__syncthreads();
    cache[tid] += cache[tid + 2];
    //__syncthreads();
    cache[tid] += cache[tid + 1];
    //__syncthreads();
}

template <size_t blockSize, typename T>
__global__ void reducebase4(T* g_idata, T* g_odata, size_t size) {
    __shared__ T sdata[blockSize];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    sdata[tid] = 0;

    T adder = i + blockDim.x < size ? g_idata[i + blockDim.x] : 0;

    if (i < size) sdata[tid] = g_idata[i] + adder;
    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid < 32) warpReduce4(sdata, tid);
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

int main() {
    // Array size
    int N = 8192 * 256;
    const size_t blockSize = 256;
    size_t bytes = N * sizeof(float);

    // Host vectors
    vector<float> h_in(N);
    float h_out_final = 0;

    cout << "Step1: h_in generation \n";
    // Initialize array with random small floats
    generate(h_in.begin(), h_in.end(), []() { return (float)(rand() % 10); });

    cout << "Step2: Mem Allocation on device \n";
    float* d_in, * d_out;
    cudaMalloc(&d_in, bytes);
    // We need a second buffer for the partial sums
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

    // --- RECURSIVE REDUCTION LOOP ---
    int current_n = N;
    while (current_n > 1) {
        // Calculate number of blocks needed
        int grid_size = (current_n + blockSize - 1) / blockSize;

        reducebase4<blockSize, float> << <grid_size, blockSize >> > (d_in, d_out, current_n);

        // After one pass, the output of this pass becomes the input for the next
        cudaMemcpy(d_in, d_out, grid_size * sizeof(float), cudaMemcpyDeviceToDevice);

        current_n = grid_size;
        if (grid_size == 1) break;
    }
    // --------------------------------

    cudaEventRecord(KernelExec, 0);

    // Copy only the single final result back
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

    // Verification
    float cpu_sum = 0;
    for (float f : h_in) cpu_sum += f;

    printf("GPU Result: %f\n", h_out_final);
    printf("CPU Result: %f\n", cpu_sum);

    if (abs(cpu_sum - h_out_final) < 1e-2) {
        cout << "COMPLETED SUCCESSFULLY\n";
    }
    else {
        cout << "VERIFICATION FAILED\n";
    }

    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}