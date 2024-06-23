#include "ray_cuda_headers.h"
MemSize query_device()
{
  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);

  if (deviceCount == 0)
  {
    printf("No CUDA support device found");
  }

  int devNo = 0;
  cudaDeviceProp iProp;
  cudaGetDeviceProperties(&iProp, devNo);

  printf("Device %d: %s \n", devNo, iProp.name);
  printf("Number of multiprocessors:                          %d\n", iProp.multiProcessorCount);
  printf("Clock rate:                                         %d\n", iProp.clockRate);
  printf("Compute capability:                                 %d.%d\n", iProp.major, iProp.minor);
  printf("Total amount of global memory:                      %4.2f KB\n", iProp.totalGlobalMem / 1024.0);
  printf("Total amount of constant memory:                    %4.2f KB\n", iProp.totalConstMem / 1024.0);
  printf("Total floats that fit in const:                     %4.2f \n", iProp.totalConstMem / (float)sizeof(float));
  //printf("Total amount of shared memory per block:          %4.2f KB\n", iProp.sharedMemPerBlock / 1024.0);
  //printf("Total amount of shared memory per multiprocessor: %4.2f KB\n", iProp.sharedMemPerMultiprocessor / 1024.0);
  //printf("Total number of registers available per block:    %4.2f KB\n", iProp.regsPerBlock / 1024.0);
  //printf("Warp size:                      %d\n", iProp.warpSize);
  printf("Maximum number of threads per block:                %d\n", iProp.maxThreadsPerBlock);
  printf("Maximum number of threads per multiprocessor:       %d\n", iProp.maxThreadsPerMultiProcessor);
  printf("Maximum grid size:                                  (%d, %d, %d)\n", iProp.maxGridSize[0], iProp.maxGridSize[1],iProp.maxGridSize[2]);
  printf("Maximum block dimension:                            (%d, %d, %d)\n", iProp.maxThreadsDim[0], iProp.maxThreadsDim[1], iProp.maxThreadsDim[2]);
  //printf("Concurrent Kernels:             %d\n", iProp.concurrentKernels);
  //printf("deviceOverlap:              %d\n", iProp.deviceOverlap);
  //printf("Memory Bus Width:             %d\n", iProp.memoryBusWidth);
  //printf("Memory Clock Rate:              %d\n", iProp.memoryClockRate);
  printf("*********************\n");

  return { iProp.totalGlobalMem, iProp.totalConstMem-936 }; // We have some kind of limit to const mem
}
