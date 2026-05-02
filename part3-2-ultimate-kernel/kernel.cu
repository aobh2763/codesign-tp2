/*
 * kernel_optimal.cu  —  Maximum-performance parallel reduction
 * N = 8192 * 256 = 2,097,152 floats
 *
 * Techniques combined from all 8 kernels:
 *  K3  — Sequential addressing          → no shared-memory bank conflicts
 *  K4  — First add during global load   → halves active threads immediately
 *  K6  — Compile-time unrolled tree     → zero runtime branch overhead
 *  K7  — __shfl_down_sync warp reduce   → stays in registers, no shared-mem reads
 *  K8  — Grid-stride loop               → each thread sums many elements → far
 *                                          fewer blocks needed, one kernel pass
 *  +   — float4 vectorised global loads → 128-bit transactions, peak bandwidth
 *  +   — __restrict__ / read-only cache → compiler emits LDG instructions
 *  +   — pointer swap instead of D2D    → eliminates inter-pass cudaMemcpy
 *  +   — __forceinline__                → no call overhead on device helpers
 */

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <stdio.h>

using std::cout;
using std::generate;
using std::vector;

// ─────────────────────────────────────────────────────────────────────────────
// 1. Warp-level reduction using shuffle instructions (K7 technique)
//    All arithmetic stays in registers — zero shared-memory traffic.
// ─────────────────────────────────────────────────────────────────────────────
template <unsigned int blockSize>
__device__ __forceinline__ float warpReduceSum(float val)
{
    // Compile-time guards → dead branches eliminated by NVCC.
    //
    // Threshold mapping (offset = blockSize / 2 rounded to next power):
    //   shfl_down(16) fires when blockSize >= 32  (combines upper/lower 16 lanes)
    //   shfl_down( 8) fires when blockSize >= 16
    //   shfl_down( 4) fires when blockSize >=  8
    //   shfl_down( 2) fires when blockSize >=  4
    //   shfl_down( 1) fires when blockSize >=  2
    if (blockSize >= 32) val += __shfl_down_sync(0xffffffff, val, 16);
    if (blockSize >= 16) val += __shfl_down_sync(0xffffffff, val, 8);
    if (blockSize >= 8)  val += __shfl_down_sync(0xffffffff, val, 4);
    if (blockSize >= 4)  val += __shfl_down_sync(0xffffffff, val, 2);
    if (blockSize >= 2)  val += __shfl_down_sync(0xffffffff, val, 1);
    return val;
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Main kernel
// ─────────────────────────────────────────────────────────────────────────────
template <unsigned int blockSize>
__global__ void reduceOptimal(
    const float* __restrict__ g_idata,   // read-only → LDG (L1 read-only cache)
    float* __restrict__ g_odata,
    unsigned int n)
{
    const unsigned int tid = threadIdx.x;
    const unsigned int laneId = tid & 31u;           // tid % 32
    const unsigned int warpId = tid >> 5u;           // tid / 32

    // ── Phase 1: grid-stride accumulation with float4 vectorised loads ────────
    //   Each thread accumulates multiple elements → single kernel pass suffices.
    //   128-bit float4 loads maximise global-memory bandwidth (K8 + vectorise).
    float mySum = 0.0f;

    // Process the bulk using 128-bit (float4) loads
    const float4* g4 = reinterpret_cast<const float4*>(g_idata);
    unsigned int   n4 = n >> 2u;                      // n / 4

    // Grid-stride: start position and hop size both in float4 units
    for (unsigned int i = blockIdx.x * blockSize + tid;
        i < n4;
        i += gridDim.x * blockSize)
    {
        float4 v = g4[i];
        mySum += v.x + v.y + v.z + v.w;
    }

    // Handle remaining elements (n not divisible by 4)
    for (unsigned int i = (n4 << 2u) + blockIdx.x * blockSize + tid;
        i < n;
        i += gridDim.x * blockSize)
    {
        mySum += g_idata[i];
    }

    // ── Phase 2: intra-warp reduction via shuffle (K7) ────────────────────────
    //   Compile-time unrolled (K6), purely register-based, no __syncthreads().
    mySum = warpReduceSum<blockSize>(mySum);

    // ── Phase 3: inter-warp reduction via tiny shared memory ─────────────────
    //   Only (blockSize/32) words — far smaller than the full sdata[blockSize]
    //   used in earlier kernels (K3–K6).  Sequential addressing (K3) means no
    //   bank conflicts.
    __shared__ float warpSums[blockSize / 32];   // e.g., 8 words for blockSize=256

    if (laneId == 0)
        warpSums[warpId] = mySum;
    __syncthreads();

    // Let the first warp do a final shuffle reduction over the warp partial sums
    if (warpId == 0) {
        mySum = (laneId < (blockSize / 32)) ? warpSums[laneId] : 0.0f;
        mySum = warpReduceSum<blockSize / 32>(mySum);
    }

    // ── Phase 4: write block result ───────────────────────────────────────────
    if (tid == 0)
        g_odata[blockIdx.x] = mySum;
}

// ─────────────────────────────────────────────────────────────────────────────
// Host driver
// ─────────────────────────────────────────────────────────────────────────────
int main()
{
    const int          N = 8192 * 256;   // 2,097,152 elements
    const unsigned int BLOCK = 256;          // threads per block
    const size_t       bytes = N * sizeof(float);

    // ── Host data ─────────────────────────────────────────────────────────────
    vector<float> h_in(N);
    float         h_out_final = 0.0f;

    cout << "Step 1: Generating input array\n";
    generate(h_in.begin(), h_in.end(), []() { return (float)(rand() % 10); });

    // ── Device allocation ─────────────────────────────────────────────────────
    cout << "Step 2: Allocating device memory\n";
    float* d_a, * d_b;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);

    // ── Timing events ─────────────────────────────────────────────────────────
    cout << "Step 3: Setting up timing events\n";
    cudaEvent_t ev_start, ev_h2d, ev_kernel, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_h2d);
    cudaEventCreate(&ev_kernel);
    cudaEventCreate(&ev_stop);

    cudaEventRecord(ev_start);

    // ── H2D transfer ──────────────────────────────────────────────────────────
    cout << "Step 4: Copying data to device\n";
    cudaMemcpy(d_a, h_in.data(), bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(ev_h2d);

    // ── Reduction loop ────────────────────────────────────────────────────────
    //   Thanks to the grid-stride loop the first pass uses only ~1024 blocks
    //   regardless of N — each block sums ~2048 elements.  The second pass
    //   trivially reduces those 1024 partial sums in a single block launch.
    //   No D2D cudaMemcpy between passes (pointer swap, K8 technique).
    cout << "Step 5: Running optimised reduction\n";
    unsigned int current_n = (unsigned int)N;

    // Choose a grid that keeps the GPU fully occupied but limits extra passes
    // Saturate ~2x SMs (conservative) — any occupancy ≥ 1 works.
    int smCount = 0;
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, 0);
    // Aim for 4 resident blocks per SM as starting grid; cap at N/BLOCK
    unsigned int grid_size = (unsigned int)std::min(
        (long long)(smCount * 4),
        (long long)((current_n + BLOCK - 1) / BLOCK));

    while (current_n > 1)
    {
        // For very small tail passes shrink the grid accordingly
        unsigned int gs = std::min(grid_size, (current_n + BLOCK - 1) / BLOCK);

        reduceOptimal<BLOCK> << <gs, BLOCK >> > (d_a, d_b, current_n);

        // Pointer swap (K8): no cudaMemcpy between passes
        float* tmp = d_a;  d_a = d_b;  d_b = tmp;

        current_n = gs;
        if (gs == 1) break;

        // After the first pass the remaining work is tiny — use a minimal grid
        grid_size = (current_n + BLOCK - 1) / BLOCK;
    }

    cudaEventRecord(ev_kernel);

    // ── D2H transfer ──────────────────────────────────────────────────────────
    // Result is always in d_a thanks to pointer swap
    cudaMemcpy(&h_out_final, d_a, sizeof(float), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);

    // ── Report timing ─────────────────────────────────────────────────────────
    float t_total, t_h2d, t_kernel, t_d2h;
    cudaEventElapsedTime(&t_total, ev_start, ev_stop);
    cudaEventElapsedTime(&t_h2d, ev_start, ev_h2d);
    cudaEventElapsedTime(&t_kernel, ev_h2d, ev_kernel);
    cudaEventElapsedTime(&t_d2h, ev_kernel, ev_stop);

    printf("\n=== Optimal Kernel Results ===\n");
    printf("Time elapsed on Host To Device Transfer: %f ms.\n", t_h2d);
    printf("Time elapsed on Reduction Kernel(s):     %f ms.\n", t_kernel);
    printf("Time elapsed on Device To Host Transfer: %f ms.\n", t_d2h);
    printf("Total Time:                              %f ms.\n\n", t_total);

    // ── Verification (double precision on CPU) ────────────────────────────────
    double cpu_sum = 0.0;
    for (float f : h_in) cpu_sum += (double)f;

    printf("GPU Result: %f\n", h_out_final);
    printf("CPU Result: %f\n", (float)cpu_sum);

    if (fabsf((float)cpu_sum - h_out_final) < 1.0f) {
        cout << "COMPLETED SUCCESSFULLY\n";
    }
    else {
        cout << "VERIFICATION FAILED — diff = "
            << fabsf((float)cpu_sum - h_out_final) << "\n";
    }

    // ── Cleanup ───────────────────────────────────────────────────────────────
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_h2d);
    cudaEventDestroy(ev_kernel);
    cudaEventDestroy(ev_stop);
    cudaFree(d_a);
    cudaFree(d_b);

    return 0;
}
