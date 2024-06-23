#include "lights.h"
#include <stdexcept>

LightMem::LightMem(){
  point_lights = nullptr;
  pl_count = 0;
}

LightMem::~LightMem(){
  if(point_lights != nullptr){
    delete[] point_lights;
    pl_count = 0;
  }
}

// Point light functions
void LightMem::pre_allocate_point_lights(size_t num_point){
  point_lights = new pointLight[num_point];
  pl_count = num_point;
}

void LightMem::assign_point_lights(size_t idx, pointLight light){
  if(idx >= pl_count){
    throw std::runtime_error("Attempted to assign more point ligths than allocated!");
  }
  point_lights[idx] = light;
}

pointLight* LightMem::get_gpu_mem_pointlight(){
  pointLight* d_lmem;
  CHECK_CUDA_ERROR(cudaMalloc((void **)&d_lmem, sizeof(pointLight)*pl_count));
  CHECK_CUDA_ERROR(cudaMemcpy(d_lmem, point_lights, 
        sizeof(pointLight)*pl_count, cudaMemcpyHostToDevice));
  return d_lmem;
}

__D__ Vec3 sample_point_light(pointLight& light){
  return light.pos;
}

