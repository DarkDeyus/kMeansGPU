#include "kernel.cuh"
#include "string.h"
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <iostream>
#define THREADS_IN_BLOCK 512
/*#define BLOCKS 1
#define THREADS (THREADS_IN_BLOCK*BLOCKS)*/
#define POINTS_PER_THREAD 128

using namespace std;

template<typename T>
__global__ void allocArray(T** d_array, unsigned int size1, unsigned int size2)
{
	unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (tid < size1)
		d_array[tid] = (T*)malloc(size2 * sizeof(T));
}

template<typename T>
static T** init_d_array(unsigned int size1, unsigned int size2)
{
	T** res;
	CHECK_ERRORS(cudaMalloc(&res, size1 * sizeof(T*)));
	allocArray << <size1 / THREADS_IN_BLOCK + !!(size1%THREADS_IN_BLOCK), THREADS_IN_BLOCK >> > (res, size1, size2);
	cudaDeviceSynchronize();
	return res;
}

template<typename T>
__global__ void freeArray(T** d_array, unsigned int size1)
{
	unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (tid < size1)
		free(d_array[tid]);
}

template<typename T>
static void free_d_array(T** d_array, unsigned int size1)
{
	freeArray << <size1 / THREADS_IN_BLOCK + !!(size1%THREADS_IN_BLOCK), THREADS_IN_BLOCK >> > (d_array, size1);
	cudaDeviceSynchronize();
	CHECK_ERRORS(cudaFree(d_array));
}


__global__ void kMeansThread(double *points, unsigned int number_of_centroids, unsigned int nT, unsigned int *membership, double * d_centres,
	unsigned int **centroid_size, double **new_centres, unsigned int *d, unsigned int offset = 0)
{
	extern __shared__ double centres[];
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x + offset;
	d[tid] = 0;

	for (int i = 0; i < number_of_centroids; ++i)
	{
		centroid_size[i][tid] = 0;
	}

	unsigned int tid_s = threadIdx.x;
	while (tid_s < number_of_centroids * DIMENSION)
	{
		centres[tid_s] = d_centres[tid_s];
		tid_s += blockDim.x;
	}

	for (int i = 0; i < number_of_centroids*DIMENSION; ++i)
	{
		new_centres[i][tid] = 0;
	}
	for (int i = 0; i < nT; ++i)
	{

		double dis = distance(&points[i*DIMENSION + nT * tid*DIMENSION], centres);
		//printf("%d %d %f\n", tid, __LINE__, dis);
		unsigned int c = 0;
		for (int j = 1; j < number_of_centroids; ++j)
		{
			double dis2 = distance(&points[i*DIMENSION + nT * tid*DIMENSION], &centres[j*DIMENSION]);
			//printf("%d %d %f %f\n", tid, __LINE__, dis, dis2);
			if (dis2 < dis)
			{
				dis = dis2;
				c = j;
			}
		}
		//printf("Thread %d, i = %d, dis = %f, c = %d, mem = %d\n", tid, i, dis, c, membership[i+nT*tid]);
		if (membership[i + nT * tid] != c)
		{
			++(d[tid]);
			//printf("Point %d changed from %d to %d\n", i + nT * tid, membership[i + nT * tid], c);
			membership[i + nT * tid] = c;
		}
		centroid_size[c][tid]++;
		//addTogether(new_centres[tid] + c * DIMENSION, &points[i*DIMENSION + nT * tid*DIMENSION]);
		for (int j = 0; j < DIMENSION; ++j)
		{
			new_centres[c*DIMENSION + j][tid] += points[i*DIMENSION + nT * tid * DIMENSION + j];
		}
	}
	/*printf("tid %d: d = %d\n", tid, d[tid]);
	unsigned int sum = 0;
	for(int i = 0; i < number_of_centroids; ++i)
	{
		printf("tid %u, nc: (%f, %f, %f), size %u\n", tid, new_centres[i*DIMENSION][tid], new_centres[i*DIMENSION + 1][tid], new_centres[i*DIMENSION + 2][tid], centroid_size[i][tid]);
		sum += centroid_size[i][tid];
	}
	if(sum != POINTS_PER_THREAD)
	{
		printf("sum = %u\n", sum);
	}*/
}

__global__ void kMeansSingleThread(double *points, unsigned int k, unsigned int nT, unsigned int *membership, double * d_centres,
	unsigned int **csize, double **new_centres, unsigned int *d, unsigned int offset, unsigned int how_many_points)
{
	extern __shared__ double centres[];
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x + offset;
	//printf("!!!!!!!!!tid %d: d = %d i = %d\n", tid, d[tid], 0);
	d[tid] = 0;
	for (int i = 0; i < k; ++i)
	{
		csize[i][tid] = 0;
	}

	unsigned int tid_s = threadIdx.x;
	while (tid_s < k * DIMENSION)
	{
		centres[tid_s] = d_centres[tid_s];
		tid_s += blockDim.x;
	}

	for (int i = 0; i < k*DIMENSION; ++i)
	{
		new_centres[i][tid] = 0;
	}
	for (int i = 0; i < how_many_points; ++i)
	{
		double dis = distance(&points[i*DIMENSION + nT * tid*DIMENSION], centres);
		//printf("%d %d %f\n", tid, __LINE__, dis);
		unsigned int c = 0;
		for (int j = 1; j < k; ++j)
		{
			double dis2 = distance(&points[i*DIMENSION + nT * tid*DIMENSION], &centres[j*DIMENSION]);
			//printf("%d %d %f %f\n", tid, __LINE__, dis, dis2);
			if (dis2 < dis)
			{
				dis = dis2;
				c = j;
			}
		}
		//printf("Thread %d, i = %d, dis = %f, c = %d, mem = %d\n", tid, i, dis, c, membership[i+nT*tid]);
		if (membership[i + nT * tid] != c)
		{
			d[tid] += 1;
			//printf("Point %d changed from %d to %d\n", i + nT * tid, membership[i + nT * tid], c);
			membership[i + nT * tid] = c;
		}
		csize[c][tid]++;
		//addTogether(new_centres[tid] + c * DIMENSION, &points[i*DIMENSION + nT * tid*DIMENSION]);
		for (int j = 0; j < DIMENSION; ++j)
		{
			new_centres[c*DIMENSION + j][tid] += points[i*DIMENSION + nT * tid * DIMENSION + j];
		}
	}
}

void kMeansGPU(double points[][DIMENSION], unsigned int number_of_centroids, unsigned int number_of_points, unsigned int membership[], double centres[][DIMENSION],
	double threshold)
{
	// we take first number_of_centroids points as the starting centroids

	unsigned int THREADS = number_of_points / POINTS_PER_THREAD;
	unsigned int BLOCKS = THREADS / THREADS_IN_BLOCK;
	unsigned int THREADS_IN_NOT_FULL_BLOCK = THREADS - BLOCKS * THREADS_IN_BLOCK;
	unsigned int POINTS_IN_NOT_FULL_THREAD = number_of_points % POINTS_PER_THREAD;
	//printf("GPU: %d %d %d %d %d\n", number_of_points,THREADS, BLOCKS, THREADS_IN_NOT_FULL_BLOCK, POINTS_IN_NOT_FULL_THREAD);

	//if there are points in the not full thread, add 1
	THREADS += !!POINTS_IN_NOT_FULL_THREAD;
	for (int i = 0; i < number_of_centroids; ++i)
	{
		for (int j = 0; j < DIMENSION; ++j)
		{
			centres[i][j] = points[i][j];
		}
	}

	double **d_new_centres, *d_centres, **d_new_centres_h;
	unsigned int *d_membership, **d_centroid_size, *d_number_of_changes, **d_centroid_size_h;

	CHECK_ERRORS(cudaMalloc(&d_membership, sizeof(unsigned int) * number_of_points));
	CHECK_ERRORS(cudaMemset(d_membership, ~0, sizeof(unsigned int) * number_of_points));
	CHECK_ERRORS(cudaMalloc(&d_centres, number_of_centroids * DIMENSION * sizeof(double)));
	CHECK_ERRORS(cudaMalloc(&d_number_of_changes, sizeof(unsigned int) * THREADS));

	d_new_centres = init_d_array<double>(number_of_centroids * DIMENSION, THREADS);
	d_new_centres_h = new double*[number_of_centroids*DIMENSION];
	CHECK_ERRORS(cudaMemcpy(d_new_centres_h, d_new_centres, sizeof(double*)*number_of_centroids*DIMENSION, cudaMemcpyDeviceToHost));
	
	d_centroid_size = init_d_array<unsigned int>(number_of_centroids, THREADS);
	d_centroid_size_h = new unsigned int*[number_of_centroids];
	CHECK_ERRORS(cudaMemcpy(d_centroid_size_h, d_centroid_size, sizeof(unsigned int*)*number_of_centroids, cudaMemcpyDeviceToHost));
	
	unsigned int number_of_changes;
	double *d_points;

	CHECK_ERRORS(cudaMalloc(&d_points, DIMENSION * number_of_points * sizeof(double)));
	CHECK_ERRORS(cudaMemcpy(d_points, points, DIMENSION * number_of_points * sizeof(double), cudaMemcpyHostToDevice));
	
	unsigned int iteration = 0;
	do
	{
		number_of_changes = 0;
		CHECK_ERRORS(cudaMemcpy(d_centres, centres, number_of_centroids * DIMENSION * sizeof(double), cudaMemcpyHostToDevice));
		if (BLOCKS)
		{			
			kMeansThread <<< BLOCKS, THREADS_IN_BLOCK, number_of_centroids * DIMENSION * sizeof(double) >> > (d_points, number_of_centroids, POINTS_PER_THREAD,
					d_membership, d_centres, d_centroid_size, d_new_centres, d_number_of_changes, 0);
			CHECK_ERRORS(cudaDeviceSynchronize());
		}


		if (THREADS_IN_NOT_FULL_BLOCK)
		{			
			kMeansThread << < 1, THREADS_IN_NOT_FULL_BLOCK, number_of_centroids * DIMENSION * sizeof(double) >> > (d_points, number_of_centroids, POINTS_PER_THREAD,
					d_membership, d_centres, d_centroid_size, d_new_centres, d_number_of_changes, THREADS_IN_BLOCK * BLOCKS);
			CHECK_ERRORS(cudaDeviceSynchronize());			
		}

		if (POINTS_IN_NOT_FULL_THREAD)
		{
			kMeansSingleThread <<< 1, 1, number_of_centroids*DIMENSION * sizeof(double) >> > (d_points, number_of_centroids, POINTS_PER_THREAD, d_membership, d_centres, d_centroid_size,
				d_new_centres, d_number_of_changes, THREADS_IN_BLOCK*BLOCKS + THREADS_IN_NOT_FULL_BLOCK, POINTS_IN_NOT_FULL_THREAD);
			CHECK_ERRORS(cudaDeviceSynchronize());
		}


		for (int i = 0; i < number_of_centroids; ++i)
		{
			thrust::device_ptr<unsigned int> d_1 = thrust::device_pointer_cast(d_centroid_size_h[i]);
			unsigned int quotient = thrust::reduce(d_1, d_1 + THREADS);
			
			//quotient = quotient == 0 ? 1 : quotient;
			//printf("i = %d, quotient = %u\n", i, quotient);
			for (int j = 0; j < DIMENSION; ++j)
			{
				thrust::device_ptr<double> d_2 = thrust::device_pointer_cast(d_new_centres_h[i * DIMENSION + j]);
				centres[i][j] = thrust::reduce(d_2, d_2 + THREADS) / quotient;

				//centres[i][j] = sumArray(d_new_centres_h[i*DIMENSION + j], THREADS) / quotient;
			}
			//printf("center of %d is : %f, %f, %f\n", i, centres[i][0], centres[i][1], centres[i][2]);
		}
		thrust::device_ptr<unsigned int> d_3 = thrust::device_pointer_cast(d_number_of_changes);
		number_of_changes = thrust::reduce(d_3, d_3 + THREADS);
		
		//printf("End of iteration (i = %d, d = %u)\n", iteration , d);
		++iteration;

	} while (number_of_changes / (float)number_of_points > threshold);

	printf("Iterations: %u", number_of_changes);

	CHECK_ERRORS(cudaMemcpy(membership, d_membership, sizeof(unsigned int)*number_of_points, cudaMemcpyDeviceToHost));

	CHECK_ERRORS(cudaFree(d_number_of_changes));
	CHECK_ERRORS(cudaFree(d_membership));
	CHECK_ERRORS(cudaFree(d_points));

	free_d_array(d_new_centres, number_of_centroids * DIMENSION);
	free_d_array(d_centroid_size, number_of_centroids);
}
