#include "kernel.cuh"
#include "kMeansCPU.h"
#include <iostream>
#include <string>
#include <fstream>
#include <iomanip>
#include <cstdio>
#include <ctime>
#include <chrono>

using namespace std;
class PointReader
{
	ifstream file;
	unsigned long long N;
	int dimension;

public:
	PointReader(string filename)
	{
		file.open(filename);
		file >> N;
		file >> dimension;
	}

	int GetDimension()
	{
		return dimension;
	}

	unsigned long long GetCount()
	{
		return N;
	}

	// [x0 y0 z0 x1 y1 z1 .... ]
	double* GetPoints()
	{
		double *points = new double[dimension * N];
		for (unsigned long long i = 0; i < N; ++i)
		{
			for (int j = 0; j < dimension; ++j)
			{
				file >> points[i * dimension + j];
			}
		}
		return points;
	}

	~PointReader()
	{
		file.close();
	}
};

void WriteResults(double* results, int number_of_centres, int dimension)
{
	for (int i = 0; i < number_of_centres; i++)
	{
		for (int j = 0; j < dimension; j++)
		{
			cout << setw(8) << setprecision(5) << results[i * dimension + j] << '\t';
		}
		cout << endl;
	}
}

void CentresToFile(string filename, double centres[][DIMENSION], int dimension, int n)
{
	ofstream stream;
	stream.open(filename);

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < dimension; j++)
		{
			stream << centres[i][j] << " ";
		}
		stream << endl;
	}

	stream.close();
}

void PointMembershipToFile(string filename, unsigned int* membership, unsigned long long n)
{
	ofstream stream;
	stream.open(filename);
	for (unsigned long long i = 0; i < n; i++)
	{
		stream << membership[i] << endl;
	}

	stream.close();
}

int main()
{
	//Opening file with points, its struct is one line with number of points and the dimension of them, and then in each line coordinates of the point, separated by spaces
	printf("Please input a filename to open:\n");
	string filename;
	cin >> filename;
	PointReader input(filename);

	double *points = input.GetPoints();
	unsigned int n = (unsigned int)input.GetCount();
	printf("Read %d of points!\n", n);

	unsigned int *membership = new unsigned int[n];

	unsigned int k;
	printf("Please input a number of centroids:");
	cin >> k;
	unsigned int dimension = input.GetDimension();
	double *centres = new double[dimension * k];

	printf("Computation on CPU in progress...\n");

	//starting timer for CPU
	auto timer = std::chrono::system_clock::now();

	kMeansCPU((double(*)[3])points, k, n, membership, (double(*)[3])centres);

	std::chrono::duration<double> duration = (std::chrono::system_clock::now() - timer);
	double timeCPU = duration.count();
	printf("CPU time was %f seconds\n", timeCPU);

	WriteResults(centres, k, dimension);
	printf("-------------------------------------------------------------------\n");
	//starting timer for GPU
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventRecord(start);


	printf("Starting counting on GPU\n");
	kMeansGPU((double(*)[3])points, k, n, membership, (double(*)[3])centres);


	cudaEventCreate(&stop);
	cudaEventRecord(stop);
	cudaEventSynchronize(start);
	cudaEventSynchronize(stop);

	float time;
	cudaEventElapsedTime(&time, start, stop);

	double timeGPU = time * 0.001;
	printf("GPU time was %f seconds\n", timeGPU);

	WriteResults(centres, k, dimension);
	
	printf("Saving results to file, please wait...\n");
	PointMembershipToFile("membership.txt", membership, n);
	CentresToFile("centres.txt", (double(*)[3])centres, dimension, k);
	
	printf("Done!\n");
	return 0;
}