#ifndef _CONST_MEMORY_MANAGER_H_
#define _CONST_MEMORY_MANAGER_H_

// This class will manage SHAPE_MEMORY for passing to the gpu.
#include "memory_manager.h"
#include <iostream>
#include <stdexcept>

using std::cout;
using std::endl;

//__constant__ float CONST_SHAPE_MEM[MAX_FLOATS_IN_CONST_MEM];

template <typename T> 
class ConstMemoryManager : public MemoryManager<T>{

public:

  ConstMemoryManager(size_t num_shapes) : 
    MemoryManager<T>(num_shapes, MAX_FLOATS_IN_CONST_MEM*sizeof(float)){}

  ConstMemoryManager(size_t num_shapes, size_t max_size) : 
    MemoryManager<T>(num_shapes, max_size){}

  void get_current_group_gpu(int* shape_count);
  void copy_from_gpu_to_group(int* shape_count);

  inline T* alloc_gpu_shapemem(){ return nullptr; }
  inline void free_gpu_shapemem(T* mem){};

};


// Get the device memory for this group, set the pointer to the next group
// Must not be null
/*template <typename T> 
void ConstMemoryManager<T>::get_current_group_gpu(int* shape_count){
  T* hmem = this->get_current_group(shape_count);

  CHECK_CUDA_ERROR(cudaMemcpyToSymbol(CONST_SHAPE_MEM, hmem, 
        (size_t)(*shape_count)*sizeof(T)));
}*/

template <typename T>
void ConstMemoryManager<T>::copy_from_gpu_to_group(int* shape_count){
  throw std::runtime_error("Copying from gpu with const memory manager not allowed.");
}

#endif
