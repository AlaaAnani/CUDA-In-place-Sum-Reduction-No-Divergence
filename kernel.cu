#include "cuda_runtime.h"
#include <cuda.h>
#include <time.h>
#include <ctime>
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include <iomanip>
using namespace std;
#define ull unsigned long long
#define SHARED_MEM_ARR_SIZE 1024
#define RAND_BOUND 1
float* A;
float* B;
double KernelTime = 0;

cudaError_t sumReductionWithCudaQ6(float* A, ull size);
int Q6Main();
__global__ void sumReductionKernelNoDivergence(float* A, ull size)
{
	__shared__ float partialSumArr[SHARED_MEM_ARR_SIZE];
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int tx = threadIdx.x;
	if (i < size)
	{
		//each thread loads one el from global memory
		partialSumArr[tx] = A[i];
		for (unsigned int stride = blockDim.x/2; stride >=1; stride = stride >>1)
		{
			__syncthreads();
			if (tx <stride && (i + stride < size))
				partialSumArr[tx] += partialSumArr[tx + stride];
		}
		__syncthreads();

		if (tx == 0)
			A[blockIdx.x] = partialSumArr[0];
	}
}
double CPUSequentialSum(float* a, int size);

int main()
{
	return Q6Main();
}

int Q6Main()
{
	std::cout << std::fixed;
	std::cout << std::setprecision(6);
	ull size;
	/* initialize random seed: */
	srand(time(NULL));
	cout << "Enter size " << endl;
	cin >> size;
	clock_t cpu_start1 = clock();

	A = (float*)malloc(sizeof(float) * size);
	for (int i = 0; i < size; i++)
		A[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / RAND_BOUND));

	//SEQUENTIAL
	clock_t cpu_start = clock();
	float SeqSum = CPUSequentialSum(A, size);
	clock_t cpu_end = clock();
	double cpu_time = (double)(cpu_end - cpu_start) / (double)CLOCKS_PER_SEC;
	double cpu_time_with_mmem = (double)(cpu_end - cpu_start1) / (double)CLOCKS_PER_SEC;
	clock_t gpu_start = clock();
	cudaError_t cudaStatus = sumReductionWithCudaQ6(A, size);
	double gpu_full_time = (double)(clock() - cpu_start) / (double)CLOCKS_PER_SEC;

	cout << "Sequential CPU Sum " << SeqSum << endl;
	cout << "Parallel GPU Sum = " << A[0] << " " << endl;
	cout << "Sequential CPU time without mem = " << cpu_time;
	cout << "	GPU Kernel time = " << KernelTime << endl;
	cout << "Sequential CPU time with mem =" << cpu_time_with_mmem;
	cout << "	GPU Full time = " << gpu_full_time << endl;
	cout << "---------\n Speedup without memory time = " << cpu_time / KernelTime << "	Speedup with memory time = " << cpu_time_with_mmem / gpu_full_time << endl;

	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "sumReductionWithCuda failed!");
		return 1;
	}

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	free(A);
	return 0;
}
cudaError_t sumReductionWithCudaQ6(float* A, ull size)
{
	float* dev_A = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for the array
	cudaStatus = cudaMalloc((void**)&dev_A, size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}


	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_A, A, size * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}


	int blockDimX = ceil(float(size) / 1024.0);
	dim3 dimBlock(1024, 1, 1);
	dim3 dimGrid(blockDimX, 1, 1);

	clock_t start = clock();
	sumReductionKernelNoDivergence <<< dimGrid, dimBlock >> > (dev_A, size);
	cudaStatus = cudaDeviceSynchronize();
	while (blockDimX > 1)
	{
		size = blockDimX;
		dim3 dimGrid(blockDimX, 1, 1);
		sumReductionKernelNoDivergence << < dimGrid, dimBlock >> > (dev_A, size);
		cudaStatus = cudaDeviceSynchronize();
		blockDimX = ceil(float(blockDimX) / 1024.0);

	}
	 KernelTime = (double)(clock() - start) / CLOCKS_PER_SEC;

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	cudaStatus = cudaMemcpy(A, dev_A, size * sizeof(float), cudaMemcpyDeviceToHost);

	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_A);

	return cudaStatus;
}
double CPUSequentialSum(float* a, int size)
{
	double sum = 0;
	for (int i = 0; i < size; i++)
		sum +=a[i];

	return sum;
}



