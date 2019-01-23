#ifndef _KMEANS_CPU_H
#define _KMEANS_CPU_H
#include "helper.cuh"

void kMeansCPU(double points[][DIMENSION], unsigned int number_of_centroids, unsigned int number_of_points, unsigned int membership[], double centres[][DIMENSION], 
	double threshold = 0.001);
#endif
