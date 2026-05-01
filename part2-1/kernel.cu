#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h" // The cuBLAS header
#include <stdio.h> 

using std::cout;
using std::generate;
using std::vector;

int main() {
	// Matrix size of N x N;
	int N = 8192;

	// Size (in bytes) of matrix
	size_t bytes = N * N * sizeof(float);

	//Nbr of Floating Operations
	float Nbr_GFLOPS;
	Nbr_GFLOPS = 2.0 * N / 1000.0 * N / 1000.0 * N / 1000.0;

	// Host vectors
	vector<float> h_a(N * N);
	vector<float> h_b(N * N);
	vector<float> h_c(N * N);

	cout << "Step1 : h_a and h_b generation \n";

	// Initialize matrices
	generate(h_a.begin(), h_a.end(), []() { return (float)(rand() % 100); });
	generate(h_b.begin(), h_b.end(), []() { return (float)(rand() % 100); });

	cout << "Step2 : Mem Allocation on host \n";
	// Allocate device memory
	float* d_a, * d_b, * d_c;
	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);
	cudaMalloc(&d_c, bytes);

	cout << "Step3 : Initialize cuBLAS and measure Time \n";

	// Create cuBLAS handle
	cublasHandle_t handle;
	cublasCreate(&handle);

	float Total_gpu_time, Host2Dev_time, Kernel_time, Dev2Host_time;
	cudaEvent_t start, stop, Host2dev, KernelExec;

	cudaEventCreate(&start);
	cudaEventCreate(&Host2dev);
	cudaEventCreate(&KernelExec);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);

	// Copy data to the device
	cout << "Step4 : Copy Data To Device \n";
	cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

	cudaEventRecord(Host2dev, 0);

	// Launch cuBLAS Kernel
	cout << "Running cuBLAS matrix multiplication... \n";
	const float alpha = 1.0f;
	const float beta = 0.0f;

	/* Note: cuBLAS is Column-Major. To get Row-Major result:
	   C = A * B  =>  C^T = B^T * A^T
	   By passing (B, A), we get the correct Row-Major C.
	*/
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
		N, N, N,
		&alpha,
		d_b, N,
		d_a, N,
		&beta,
		d_c, N);

	// record time after kernel execution
	cudaEventRecord(KernelExec, 0);

	// Copy back to the host
	cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	// compute time elapse
	cudaEventElapsedTime(&Total_gpu_time, start, stop);
	cudaEventElapsedTime(&Host2Dev_time, start, Host2dev);
	cudaEventElapsedTime(&Kernel_time, Host2dev, KernelExec);
	cudaEventElapsedTime(&Dev2Host_time, KernelExec, stop);

	printf("Time elapsed on Host To Device Transfer: %f ms.\n\n", Host2Dev_time);
	printf("Time elapsed on matrix multiplication on GPU (cuBLAS): %f ms.\n\n", Kernel_time);
	printf("Time elapsed on Device To Host Transfer: %f ms.\n\n", Dev2Host_time);
	printf("Total Time: %f ms.\n\n", Total_gpu_time);

	float Perf_GFLOPS;
	Perf_GFLOPS = Nbr_GFLOPS * 1000 / Kernel_time;
	printf("Kernel Execution Performance: %f GFLOPS.\n\n", Perf_GFLOPS);

	cout << "COMPLETED SUCCESSFULLY\n";

	// Free memory
	cublasDestroy(handle);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	//wait for keyboard press
	int kml;
	scanf("%c", &kml);

	return 0;
}