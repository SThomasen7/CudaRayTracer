#include "material_mem.h"
#include <exception>
#include <iostream>

using std::cout;
using std::endl;


MaterialMem::MaterialMem(size_t num_materials){

  if(_NUM_MATERIALS_ != 0){
    throw std::runtime_error("_NUM_MATERIALS_ ALREADY SET, DID YOU CREATE TWO MATERIALMEM CLASSES?");
  }

  _NUM_MATERIALS_ = num_materials;
  this->num_materials = num_materials;

  if(_NUM_MATERIALS_ >= (MATERIAL_MEM_LEN / (sizeof(Material) / sizeof(float)))){
    throw std::runtime_error("Too many materials allocated for memory to handle!");
  }

}

void MaterialMem::check_in_range(size_t idx){
  if(idx >= _NUM_MATERIALS_){
    throw std::runtime_error("Material idx out of range!");
  }
}

void MaterialMem::set_material(size_t idx, Material mat){
  check_in_range(idx);

  float* material_spot = &data[(sizeof(Material)/sizeof(float))*idx];

  material_spot[0] = mat.color.x;
  material_spot[1] = mat.color.y;
  material_spot[2] = mat.color.z;

  material_spot[3] = mat.prob_reflect;
  material_spot[4] = mat.prob_refract;
  material_spot[5] = mat.fuzz;
}

void MaterialMem::set_metal(size_t idx, Vec3 color, float fuzz){
  set_material(idx, {color, 1.0f, 0.0f, fuzz});
}

void MaterialMem::set_metal(size_t idx){
  set_metal(idx, {1.0f, 1.0f, 1.0f}, 0.0f);
}

void MaterialMem::set_glass(size_t idx, Vec3 color, float fuzz){
  set_material(idx, {color, 0.0f, 1.0f, fuzz});
}


void MaterialMem::set_glass(size_t idx){ // Perfect refract
  set_glass(idx, {1.0f, 1.0f, 1.0f}, 0.0f);
}

void MaterialMem::set_diffuse(size_t idx, Vec3 color){
  set_material(idx, {color, 0.0f, 0.0f, 1.0f});
}

__D__ Material get_material(float* mem, size_t idx){
  const size_t sizeof_material = 6;
  size_t matid = idx*(sizeof_material);
  //size_t matid = 0;
  return {
    {mem[matid+0], mem[matid+1], mem[matid+2]},
    mem[matid+3],
    mem[matid+4],
    mem[matid+5]
  };
}

float* MaterialMem::get_material_gpu_mem(){
  float* d_mem;
  //cout << sizeof(Material) << " " << num_materials << " " << endl;
  CHECK_CUDA_ERROR(cudaMalloc((void **)&d_mem, sizeof(Material)*num_materials));
  CHECK_CUDA_ERROR(cudaMemcpy(d_mem, &data[0], sizeof(Material)*num_materials,
        cudaMemcpyHostToDevice));
  return d_mem;
}


void MaterialMem::free_material_gpu_mem(float* dmem){
  cudaFree(dmem);
}
