#ifndef _SPHERE_H_
#define _SPHERE_H_ 1

#include "ray_cuda_headers.h"
#include "material.h"
#include "vec3.h"
#include "memory_manager.h"
#include "hit.h"
#include "ray.h"

typedef struct Sphere{
  Vec3 center;
  float radius;
  float material_index;
} Sphere;

/*class SphereMemoryManager : public MemoryManager<float>{

public:

  SphereMemoryManager(size_t num_spheres, size_t mem_size_bytes);
  SphereMemoryManager(size_t num_spheres);

  void set_sphere(int index, Vec3 center, float radius,
      int material_index);
  void set_sphere(int index, Sphere sphere);

  // Mem options
  void get_current_group_gpu(float* dmem, int* shape_count);

};*/

__D__ Sphere get_sphere(int index, float* shape_mem);
__D__ bool sphere_hit(Ray& ray, HitRecord& hit, Sphere& sphere);
__D__ bool sphere_shadow_hit(Ray& ray, HitRecord& hit, Sphere& sphere, float tlim);

#endif
