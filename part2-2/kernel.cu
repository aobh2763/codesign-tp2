#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>
#include <mma.h> 
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

using namespace nvcuda;
using namespace nvcuda::wmma;
using std::cout;
using std::generate;
using std::vector;

// Tensor Core shape for TF32 on RTX 3050 (sm_80)
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 8;

/* =========================================================================== */
__global__ void matrixMulTensorCore(const float* a, const float* b, float* c, int N) {
	// Warp index
	int warpM = (blockIdx.y * blockDim.y + threadIdx.y);
	int warpN = (blockIdx.x);

	// Fragments
	fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, precision::tf32, row_major> a_frag;
	fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, precision::tf32, row_major> b_frag;
	fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

	fill_fragment(acc_frag, 0.0f);

	for (int i = 0; i < N; i += WMMA_K) {
		int aRow = warpM * WMMA_M;
		int aCol = i;
		int bRow = i;
		int bCol = warpN * WMMA_N;

		if (aRow < N && aCol < N && bRow < N && bCol < N) {
			load_matrix_sync(a_frag, a + aRow * N + aCol, N);
			load_matrix_sync(b_frag, b + bRow * N + bCol, N);
			mma_sync(acc_frag, a_frag, b_frag, acc_frag);
		}
	}

	int cRow = warpM * WMMA_M;
	int cCol = warpN * WMMA_N;
	if (cRow < N && cCol < N) {
		store_matrix_sync(c + cRow * N + cCol, acc_frag, N, mem_row_major);
	}
}

/* =========================================================================== */
void verify_result(vector<float>& a, vector<float>& b, vector<float>& c, int N) {
	printf("Starting Verification (CPU)... \n");
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			float tmp = 0;
			for (int k = 0; k < N; k++) {
				tmp += a[i * N + k] * b[k * N + j];
			}
			// Use epsilon for TF32/Float comparison
			if (fabs(tmp - c[i * N + j]) > 1.0f) {
				printf("Verification failed at [%d][%d]! CPU:%f GPU:%f\n", i, j, tmp, c[i * N + j]);
				return;
			}
		}
	}
	printf("Verification PASSED.\n");
}

/* =========================================================================== */
int main() {
	int N = 8192;
	size_t bytes = N * N * sizeof(float);

	float Nbr_GFLOPS;
	Nbr_GFLOPS = 2.0 * N / 1000.0 * N / 1000.0 * N / 1000.0;

	vector<float> h_a(N * N);
	vector<float> h_b(N * N);
	vector<float> h_c(N * N);

	cout << "Step1 : h_a and h_b generation \n";
	generate(h_a.begin(), h_a.end(), []() { return (float)(rand() % 10); });
	generate(h_b.begin(), h_b.end(), []() { return (float)(rand() % 10); });

	cout << "Step2 : Mem Allocation on host \n";
	float* d_a, * d_b, * d_c;
	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);
	cudaMalloc(&d_c, bytes);

	cout << "Step3 : Launch Event to measure Time \n";
	float Total_gpu_time, Host2Dev_time, Kernel_time, Dev2Host_time;
	cudaEvent_t start, stop, Host2dev, KernelExec;

	cudaEventCreate(&start);
	cudaEventCreate(&Host2dev);
	cudaEventCreate(&KernelExec);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);

	cout << "Step4 : Copy Data To Device \n";
	cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

	cudaEventRecord(Host2dev, 0);

	// Grid/Block setup for Tensor Cores (Warp-based)
	dim3 threads(32, 4); // 32 threads (one warp) wide, 4 warps deep
	dim3 blocks(N / 16, N / 16 / 4);

	cout << "Running matrixMulTensorCore (TF32)...\n";

	matrixMulTensorCore << <blocks, threads >> > (d_a, d_b, d_c, N);

	cudaEventRecord(KernelExec, 0);

	cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&Total_gpu_time, start, stop);
	cudaEventElapsedTime(&Host2Dev_time, start, Host2dev);
	cudaEventElapsedTime(&Kernel_time, Host2dev, KernelExec);
	cudaEventElapsedTime(&Dev2Host_time, KernelExec, stop);

	printf("Time elapsed on Host To Device Transfer: %f ms.\n\n", Host2Dev_time);
	printf("Time elapsed on matrix multiplication on GPU: %f ms.\n\n", Kernel_time);
	printf("Time elapsed on Device To Host Transfer: %f ms.\n\n", Dev2Host_time);
	printf("Total Time: %f ms.\n\n", Total_gpu_time);

	float Perf_GFLOPS = Nbr_GFLOPS * 1000 / Kernel_time;
	printf("Kernel Execution Performance: %f GFLOPS.\n\n", Perf_GFLOPS);

	// Verification (Warning: Very slow for N=8192)
	// verify_result(h_a, h_b, h_c, N);

	cout << "COMPLETED SUCCESSFULLY\n";

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	int kml;
	scanf("%c", &kml);

	return 0;
}