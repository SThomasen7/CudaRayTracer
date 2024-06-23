#ifndef _MATERIAL_MEM_H_
#define _MATERIAL_MEM_H_ 1

#include "ray_cuda_headers.h"
#include "material.h"
#include "vec3.h"

inline size_t _NUM_MATERIALS_ = 0;

// Note that this does not have anything to do with
// memory manager class
//
// Singleton class
class MaterialMem{
public:
  MaterialMem(size_t num_materials);

  void set_material(size_t idx, Material mat);

  void set_metal(size_t idx, Vec3 color, float fuzz);
  void set_metal(size_t idx); // Perfect reflect

  void set_glass(size_t idx, Vec3 color, float fuzz);
  void set_glass(size_t idx); // Perfect refract
                             
  void set_diffuse(size_t idx, Vec3 color);

  float* get_material_gpu_mem();
  void free_material_gpu_mem(float*);

private:

  void check_in_range(size_t);
  float data[MATERIAL_MEM_LEN];
  size_t num_materials;
};

// The material form the device is read only.
__D__ Material get_material(float* mem, size_t idx);

#endif
