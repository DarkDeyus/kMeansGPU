#include "helper.cuh"

__host__ __device__ double distance(double * v1, double * v2)
{
	double sum = 0;
	for (int i = 0; i < DIMENSION; ++i)
	{
		sum += (v1[i] - v2[i]) * (v1[i] - v2[i]);
	}
	return sum;
}

__host__ __device__ void addTogether(double * v1, double * v2)
{
	for (int i = 0; i < DIMENSION; ++i)
	{
		v1[i] += v2[i];
	}
}