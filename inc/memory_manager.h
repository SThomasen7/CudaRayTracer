#ifndef _MEMORY_MANAGER_H_
#define _MEMORY_MANAGER_H_

// This class will manage SHAPE_MEMORY for passing to the gpu.
#include "vec3.h"
#include "ray_cuda_headers.h"
#include <iostream>

using std::cout;
using std::endl;

const size_t SHAPE_MEM_GROUP_SIZE = 844800000/2; // Max number of elements bytes group
//const size_t SHAPE_MEM_GROUP_SIZE = 1689600000; // Max number of elements bytes group

template <typename T> 
class MemoryManager{

public:

  MemoryManager(size_t num_shapes, size_t max_size);
  MemoryManager(size_t num_shapes);
  ~MemoryManager();

  size_t num_groups();
  size_t shapes_per_group();
  size_t current_group_id();
  void rewind(); // Sets the current group to 0
  bool increment_group_idx(); // Returns true when there is new data
  
  // Moves index to the next group and gets pointer to mem, 
  // sets int to how many shapes are available in group
  T* get_current_group(int* shape_count);
  // If this pointer is null, allocates device gpu mem,
  // if not, it over writes device gpu mem.
  void get_current_group_gpu(T* dmem, int* shape_count);
  void copy_from_gpu_to_group(T* dmem, int* shape_count);

  T* alloc_gpu_shapemem();
  void free_gpu_shapemem(T*);

  inline size_t get_num_shapes(){ return num_shapes; }
  void copy_to_mem_manager(T* from);
  void copy_from_mem_manager(T* to);

protected:
  T* shape_mem;      // Shape memory
  size_t num_shapes; // Total number of shapes
  size_t group_size; // Bytes per group

private:

  // Manage shape memory here
  int current_group;

};

//using std::cout;
//using std::endl;

// Init the class
template <typename T>
MemoryManager<T>::MemoryManager(size_t num_shapes, size_t group_size){
  this->group_size = group_size;
  this->num_shapes = num_shapes;
  shape_mem = (T*)malloc(num_shapes * sizeof(T));
  current_group = 0;
}

template <typename T>
MemoryManager<T>::MemoryManager(size_t num_shapes){
  this->group_size = SHAPE_MEM_GROUP_SIZE;
  this->num_shapes = num_shapes;
  shape_mem = (T*)malloc(num_shapes * sizeof(T));
  current_group = 0;
}

// Free host memory.
template <typename T>
MemoryManager<T>::~MemoryManager(){
  num_shapes = 0;
  if(shape_mem != nullptr){
    free(shape_mem);
    shape_mem = nullptr;
  }
  current_group = 0;
}

// Get the number of groups
template <typename T> 
size_t MemoryManager<T>::shapes_per_group(){
  return group_size / sizeof(T);
}

template <typename T> 
size_t MemoryManager<T>::num_groups(){
  int num_groups = ((sizeof(T)*num_shapes) / group_size);
  // If the total memory needed is not a multiple of the group size 
  // add one for overflow
  // OR if the size of the memory needed is less than the group size
  if(((num_shapes*(sizeof(T))) % group_size) != 0 || (
        num_groups == 0 && num_shapes > 0)){
    num_groups += 1;
  }
  return  num_groups;
}

// Get the current group idx
template <typename T>
size_t MemoryManager<T>::current_group_id(){
  return current_group;
}

// Set the group to the zeroth group
template <typename T>
void MemoryManager<T>::rewind(){
  current_group = 0;
}

// Get the host memory for this group
template <typename T>
T* MemoryManager<T>::get_current_group(int* shape_count){
  //*shape_count = (shapes_per_group()*(current_group+1)) % num_shapes;
  if(current_group+1 == num_groups() && num_groups() != 1){
    //cout << "A" << endl;
    *shape_count = shapes_per_group()-
                    ((shapes_per_group()*num_groups()) % num_shapes);
  } else if (num_groups() == 1){
    //cout << "B" << num_shapes << endl;
    *shape_count = num_shapes;
  } else{
    //cout << "C" << endl;
    *shape_count = shapes_per_group();
  }
  return &shape_mem[current_group*shapes_per_group()];
}

// Get the device memory for this group, set the pointer to the next group
// Must not be null
template <typename T> 
void MemoryManager<T>::get_current_group_gpu(T* dmem, int* shape_count){
  T* hmem = get_current_group(shape_count);

  // Copy hmem to dmem
  CHECK_CUDA_ERROR(cudaMemcpy(dmem, hmem, *shape_count*sizeof(T),
        cudaMemcpyHostToDevice));
}

template <typename T>
void MemoryManager<T>::copy_from_gpu_to_group(T* dmem, int* shape_count){
  T* hmem = get_current_group(shape_count);
  CHECK_CUDA_ERROR(cudaMemcpy(hmem, dmem, *shape_count*sizeof(T),
        cudaMemcpyDeviceToHost));

}

// Private, just increments the idx to the next group
template <typename T> 
bool MemoryManager<T>::increment_group_idx(){
  current_group = (current_group + 1) % num_groups();
  return current_group != 0 || num_groups() == 0;
}

// Get gpu memory
template <typename T> 
T*MemoryManager<T>::alloc_gpu_shapemem(){
  T* dmem;
  CHECK_CUDA_ERROR(cudaMalloc((void**)&dmem, group_size));
  return dmem;
}

// Free gpu memory
template <typename T>
void MemoryManager<T>::free_gpu_shapemem(T *dmem){
  CHECK_CUDA_ERROR(cudaFree(dmem));
}

template <typename T>
void MemoryManager<T>::copy_to_mem_manager(T* from){
  memcpy(shape_mem, from, sizeof(T)*num_shapes);
}

template <typename T>
void MemoryManager<T>::copy_from_mem_manager(T* to){
  memcpy(to, shape_mem, sizeof(T)*num_shapes);
}

#endif
