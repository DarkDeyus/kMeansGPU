#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <iostream>
#include "kernel.cuh"
#include "string.h"
using namespace std;

const int THREADS_IN_BLOCK = 512;

const int POINTS_PER_THREAD = 128;


//using kernels to locate and free 2 dimensional arrays

template<typename T>
__global__ void freeArray(T** d_array, unsigned int first_size)
{
	unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (tid < first_size)
		free(d_array[tid]);
}

//result is d_array[first_size][second_size]
template<typename T>
__global__ void allocArray(T** d_array, unsigned int first_size, unsigned int second_size)
{
	unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (tid < first_size)
		d_array[tid] = (T*)malloc(second_size * sizeof(T));
}

template<typename T>
static T** init_d_array(unsigned int size1, unsigned int size2)
{
	T** result;
	CHECK_ERRORS(cudaMalloc(&result, size1 * sizeof(T*)));
	//count how many blocks do we need, add one to it if something left
	allocArray << <first_size / THREADS_IN_BLOCK + !!(first_size % THREADS_IN_BLOCK), THREADS_IN_BLOCK >> > (result, first_size, second_size);
	cudaDeviceSynchronize();
	return result;
}



template<typename T>
static void free_d_array(T** d_array, unsigned int size1)
{
	freeArray << <first_size / THREADS_IN_BLOCK + !!(first_size%THREADS_IN_BLOCK), THREADS_IN_BLOCK >> > (d_array, first_size);
	cudaDeviceSynchronize();
	CHECK_ERRORS(cudaFree(d_array));
}

// Kernel used to run threads FULL of points , every thread checks full POINTS_PER_THREAD points.
__global__ void KMeansFullThreads(double *points, unsigned int number_of_centroids, unsigned int nT, unsigned int *membership, double * d_centres,
	unsigned int **centroid_size, double **new_centres, unsigned int *d, unsigned int offset = 0)
{
	extern __shared__ double centres[];
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x + offset;
	d[tid] = 0;

	for (int i = 0; i < number_of_centroids; ++i)
	{
		centroid_size[i][tid] = 0;
	}

	unsigned int tid_second = threadIdx.x;
	//in case there are less threads than fields in the shared centres array
	while (tid_second < number_of_centroids * DIMENSION)
	{
		centres[tid_second] = d_centres[tid_second];
		tid_second += blockDim.x;
	}

	for (int i = 0; i < number_of_centroids*DIMENSION; ++i)
	{
		new_centres[i][tid] = 0;
	}
	for (int i = 0; i < nT; ++i)
	{

		double dist1 = distance(&points[i * DIMENSION + nT * tid * DIMENSION], centres);

		unsigned int current_centroid_number = 0;
		for (int j = 1; j < number_of_centroids; ++j)
		{
			double dist2 = distance(&points[i * DIMENSION + nT * tid * DIMENSION], &centres[j * DIMENSION]);

			if (dist2 < dist1)
			{
				dist1 = dist2;
				current_centroid_number = j;
			}
		}

		if (membership[i + nT * tid] != current_centroid_number)
		{
			++(d[tid]);
			membership[i + nT * tid] = current_centroid_number;
		}
		centroid_size[current_centroid_number][tid]++;

		for (int j = 0; j < DIMENSION; ++j)
		{
			new_centres[current_centroid_number * DIMENSION + j][tid] += points[i * DIMENSION + nT * tid * DIMENSION + j];
		}
	}	
}

//kernel used with only one thread that checks how_many_points , not full POINTS_PER_THREAD
__global__ void KMeansSingleNotFullThread(double *points, unsigned int number_of_centroids, unsigned int nT, unsigned int *membership, double * d_centres,
	unsigned int **centroid_size, double **new_centres, unsigned int *number_of_changes, unsigned int offset, unsigned int how_many_points)
{
	extern __shared__ double centres[];
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x + offset;

	number_of_changes[tid] = 0;

	for (int i = 0; i < number_of_centroids; ++i)
	{
		centroid_size[i][tid] = 0;
	}

	unsigned int tid_second = threadIdx.x;
	//in case there are less threads than fields in the shared centres array
	while (tid_second < number_of_centroids * DIMENSION)
	{
		centres[tid_second] = d_centres[tid_second];
		tid_second += blockDim.x;
	}

	for (int i = 0; i < number_of_centroids * DIMENSION; ++i)
	{
		new_centres[i][tid] = 0;
	}
	for (int i = 0; i < how_many_points; ++i)
	{
		double dist = distance(&points[i * DIMENSION + nT * tid * DIMENSION], centres);

		unsigned int current_centroid_number = 0;
		for (int j = 1; j < number_of_centroids; ++j)
		{
			double dist2 = distance(&points[i * DIMENSION + nT * tid * DIMENSION], &centres[j * DIMENSION]);

			if (dist2 < dist)
			{
				dist = dist2;
				current_centroid_number = j;
			}
		}

		if (membership[i + nT * tid] != current_centroid_number)
		{
			number_of_changes[tid] += 1;
			membership[i + nT * tid] = current_centroid_number;
		}
		centroid_size[current_centroid_number][tid]++;
		for (int j = 0; j < DIMENSION; ++j)
		{
			new_centres[current_centroid_number * DIMENSION + j][tid] += points[i * DIMENSION + nT * tid * DIMENSION + j];
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
			KMeansFullThreads <<< BLOCKS, THREADS_IN_BLOCK, number_of_centroids * DIMENSION * sizeof(double) >> > (d_points, number_of_centroids, POINTS_PER_THREAD,
					d_membership, d_centres, d_centroid_size, d_new_centres, d_number_of_changes, 0);
			CHECK_ERRORS(cudaDeviceSynchronize());
		}


		if (THREADS_IN_NOT_FULL_BLOCK)
		{			
			KMeansFullThreads << < 1, THREADS_IN_NOT_FULL_BLOCK, number_of_centroids * DIMENSION * sizeof(double) >> > (d_points, number_of_centroids, POINTS_PER_THREAD,
					d_membership, d_centres, d_centroid_size, d_new_centres, d_number_of_changes, THREADS_IN_BLOCK * BLOCKS);
			CHECK_ERRORS(cudaDeviceSynchronize());			
		}

		if (POINTS_IN_NOT_FULL_THREAD)
		{
			KMeansSingleNotFullThread <<< 1, 1, number_of_centroids*DIMENSION * sizeof(double) >> > (d_points, number_of_centroids, POINTS_PER_THREAD, d_membership, d_centres, d_centroid_size,
				d_new_centres, d_number_of_changes, THREADS_IN_BLOCK *BLOCKS + THREADS_IN_NOT_FULL_BLOCK, POINTS_IN_NOT_FULL_THREAD);
			CHECK_ERRORS(cudaDeviceSynchronize());
		}


		for (int i = 0; i < number_of_centroids; ++i)
		{
			thrust::device_ptr<unsigned int> d_1 = thrust::device_pointer_cast(d_centroid_size_h[i]);
			unsigned int quotient = thrust::reduce(d_1, d_1 + THREADS);

			//if there is no point in a centroid , we would divide by 0, so we set it to 1
			//quotient = quotient == 0 ? 1 : quotient;
			//printf("centroid number is %d, quotient is %u\n", i, quotient);
			for (int j = 0; j < DIMENSION; ++j)
			{
				thrust::device_ptr<double> d_2 = thrust::device_pointer_cast(d_new_centres_h[i * DIMENSION + j]);
				centres[i][j] = thrust::reduce(d_2, d_2 + THREADS) / quotient;
			}
			//printf("center of %d is : %f, %f, %f\n", i, centres[i][0], centres[i][1], centres[i][2]);
		}
		thrust::device_ptr<unsigned int> d_3 = thrust::device_pointer_cast(d_number_of_changes);
		number_of_changes = thrust::reduce(d_3, d_3 + THREADS);
		
		//printf("End of iteration number %d, d is equal to %u)\n", iteration , d);
		++iteration;

	} while (number_of_changes / (float)number_of_points > threshold);

	printf("Number of iterations: %u", number_of_changes);

	CHECK_ERRORS(cudaMemcpy(membership, d_membership, sizeof(unsigned int)*number_of_points, cudaMemcpyDeviceToHost));

	CHECK_ERRORS(cudaFree(d_number_of_changes));
	CHECK_ERRORS(cudaFree(d_membership));
	CHECK_ERRORS(cudaFree(d_points));

	free_d_array(d_new_centres, number_of_centroids * DIMENSION);
	free_d_array(d_centroid_size, number_of_centroids);
}
