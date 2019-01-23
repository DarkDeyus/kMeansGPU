#include "kMeansCPU.h"
#include "string.h"

void kMeansCPU(double points[][DIMENSION], unsigned int number_of_centroids, unsigned int number_of_points, unsigned int membership[], double centres[][DIMENSION], double threshold)
{
	//First number_of_centroids points are the centroids.
	for (int i = 0; i < number_of_centroids; ++i)
	{
		for (int j = 0; j < DIMENSION; ++j)
		{
			centres[i][j] = points[i][j];
		}
	}
	unsigned int *centroid_size = new unsigned int[number_of_centroids];
	//set -1 everywhere, as in the beginning none of the points belong to any centroid
	memset(membership, ~0, sizeof(unsigned int) * number_of_points);

	double *new_centres = new double[sizeof(double) * number_of_centroids * DIMENSION ];
	unsigned int number_of_changes;
	do
	{
		number_of_changes = 0;
		memset(centroid_size, 0, sizeof(unsigned int) * number_of_centroids);
		for (int i = 0; i < number_of_centroids * DIMENSION; ++i)
		{
			new_centres[i] = 0.0f;
		}
		for (int i = 0; i < number_of_points; ++i)
		{
			double dist = distance(points[i], centres[0]);
			unsigned int new_centroid_number = 0;
			//find the centroid with the smallest distance to the current point

			for (int j = 1; j < number_of_centroids; ++j)
			{
				double dist2 = distance(points[i], centres[j]);
				if(dist2 < dist)
				{
					dist = dist2;
					new_centroid_number = j;
				}
			}

			if (membership[i] != new_centroid_number)
			{
				++number_of_changes;
				membership[i] = new_centroid_number;
			}
			centroid_size[new_centroid_number]++;
			//we are adding together all coordinates of all points that belong to the given centroid, so we can find the new centroid later on
			addTogether(new_centres + new_centroid_number * DIMENSION, points[i]);
		}
		//printf("---Sizes---\n %d %d %d %d\n", centroid_size[0], centroid_size[1], centroid_size[2], centroid_size[3]);
		//printf("---New Centres---\n");
		for (int i = 0; i < number_of_centroids; ++i)
		{
			for (int j = 0; j < DIMENSION; ++j)
			{
				//now we just divide it by the number of points in a centroid to get the new coordinates
				centres[i][j] = new_centres[i * DIMENSION + j] / centroid_size[i];
				//printf("%f ", centres[i][j]);
			}
			//printf("\n");
		}
	} while (number_of_changes / (double)number_of_points > threshold);

	delete[] centroid_size;
	delete[] new_centres;
}