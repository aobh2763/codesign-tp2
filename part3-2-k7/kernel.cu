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

// Kernel definition as provided
template <unsigned int blockSize, typename T>
__device__ __forceinline__ T warpReduceSum(T sum) {
    if (blockSize >= 32)
        sum += __shfl_down_sync(0xffffffff, sum, 16);  // 0-16, 1-17, 2-18, etc.
    if (blockSize >= 16)
        sum += __shfl_down_sync(0xffffffff, sum, 8);  // 0-8, 1-9, 2-10, etc.
    if (blockSize >= 8)
        sum += __shfl_down_sync(0xffffffff, sum, 4);  // 0-4, 1-5, 2-6, etc.
    if (blockSize >= 4)
        sum += __shfl_down_sync(0xffffffff, sum, 2);  // 0-2, 1-3, 4-6, 5-7, etc.
    if (blockSize >= 2)
        sum += __shfl_down_sync(0xffffffff, sum, 1);  // 0-1, 2-3, 4-5, etc.
    return sum;
}

template <size_t blockSize, typename T>
__global__ void reducebase6(T* g_idata, T* g_odata, size_t size) {
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    T sum = i < size ? g_idata[i] : 0;
    __syncthreads();

    // Shared mem for partial sums (one per warp in the block)
    static __shared__ T warpLevelSums[WARP_SIZE];
    const int laneId = threadIdx.x % WARP_SIZE;
    const int warpId = threadIdx.x / WARP_SIZE;

    sum = warpReduceSum<blockSize>(sum);

    if (laneId == 0) warpLevelSums[warpId] = sum;
    __syncthreads();

    // read from shared memory only if that warp existed
    sum = (threadIdx.x < blockDim.x / WARP_SIZE) ? warpLevelSums[laneId] : 0;
    // Final reduce using first warp
    if (warpId == 0) sum = warpReduceSum<blockSize / WARP_SIZE>(sum);

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sum;
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

        reducebase6<blockSize, float> << <grid_size, blockSize >> > (d_in, d_out, current_n);

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