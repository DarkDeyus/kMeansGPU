#ifndef _KERNEL_CUH
#define _KERNEL_CUH

#include "helper.cuh"

void kMeansGPU(double points[][DIMENSION], unsigned int number_of_centroids, unsigned int number_of_points, unsigned int membership[], double centres[][DIMENSION],
				double threshold = 0.001);
#endif
