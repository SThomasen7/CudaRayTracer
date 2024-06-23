#ifndef _RENDER_ENGINE_H_
#define _RENDER_ENGINE_H_ 1

#include "ray_tracing.h"
#include "consts.h"
#include "camera.h"
#include "ray_cuda_headers.h"
#include "sphere.h"
#include "const_memory_manager.h"
#include "triangle_memory_manager.h"
#include "lights.h"
#include "image.h"
#include "config.h"
#include "material_mem.h"
#include <string>

class RenderEngine{

public:

  RenderEngine(SystemConfig sconfig, MemSize mem_size);
  ~RenderEngine();

  void set_sphere_memory(MemoryManager<Sphere>* memory_manager);
  void set_triangle_memory(TriangleMemoryManager** memory_managers, size_t c);
  void set_material_memory(MaterialMem* memory_manager);
  void set_light_memory(LightMem* light_mem);
  void save_image(std::string filename);

  void render();

  void sphere_pass(ThreadMem* dmem, int thread_group, int depth);
  void triangle_pass(ThreadMem* dmem, int thread_group, int depth);
  void sphere_shadow_pass(ThreadMem* dmem, int thread_group, int depth,
                          Vec3 &light_pos);
  void triangle_shadow_pass(ThreadMem* dmem, int thread_group, int depth,
                          Vec3 &light_pos);
  void illumination_pass(ThreadMem* dmem, int thread_group, int depth);

  /* phases in illumination pass*/
  void point_light_pass(ThreadMem* dmem, int thread_group, int depth);

private:

  SystemConfig config;
  MemSize mem_size;

  Camera* camera;
  Image* img;
  MemoryManager<Sphere>* sphere_mem;
  TriangleMemoryManager** tri_mem;
  size_t triangle_mesh_c;
  LightMem* light_mem;
  MaterialMem* material_mem;

  int thread_count;
  dim3 block;
  dim3 grid;

};


#endif
