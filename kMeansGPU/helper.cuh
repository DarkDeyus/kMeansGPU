#ifndef _HELPER_CUH
#define _HELPER_CUH
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define DIMENSION 3

#define CHECK_ERRORS_FORMAT(status, format, ...) do{\
	if(cudaSuccess != status) {\
		fprintf(stderr, "Cuda Error in %s:%d - %s - ", __FILE__, __LINE__, cudaGetErrorString(status));\
		fprintf(stderr, format, __VA_ARGS__);\
		fprintf(stderr, "\n");\
	}\
}while(0)

#define CHECK_ERRORS(status) do{\
	if(cudaSuccess != status) {\
		fprintf(stderr, "Cuda Error in %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(status));\
	}\
}while(0)

__host__ __device__ double distance(double * v1, double * v2);
__host__ __device__ void addTogether(double * v1, double * v2);
#endif
