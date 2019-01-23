all:
	nvcc -G -g -O0 -std=c++11 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21  -odir "." -M -o "main.d" "main.cpp"
	nvcc -G -g -O0 -std=c++11 --compile  -x c++ -o  "main.o" "main.cpp"
	nvcc --cudart static --relocatable-device-code=true -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_21 -link -o  "KMeansLinux"  ./cudaTimer.o ./gpuSums.o ./helper.o ./kMeansCPU.o ./kernel.o ./main.o   
