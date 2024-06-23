#ifndef _RAY_CUDA_HEADERS_H_
#define _RAY_CUDA_HEADERS_H_ 1

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"
#include <iostream>

#define __D__ __device__
#define __H__ __host__
#define __HD__ __host__ __device__
#define __DC__ __device__ __constant__

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)

#define MAX_FLOATS_IN_CONST_MEM 16150
template <typename T>
void check(T err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorName(err) << std::endl;
        std::cerr << cudaGetErrorString(err) << " in func " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

typedef struct MemSize{
  size_t global_mem;
  size_t const_mem;
} MemSize;

MemSize query_device();

#endif
