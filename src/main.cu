#include "consts.h"
#include "config.h"
#include "sphere.h"
#include "render_engine.h"
#include "vec3.h"
#include "ray_cuda_headers.h"
#include "material_mem.h"
#include "stdio.h"
#include "material.h"
#include "parser.h"
#include "asset.h"
#include <iostream>

using namespace std;

int main(int argc, char** argv){

  MemSize mem_size = query_device();

  bool fail;
  SystemConfig config = parse_cl(argc, argv, fail);
  if(fail){
    return 1;
  }

  RenderEngine render_engine(config, mem_size);
  //MemoryManager<Sphere> sphere_manager(80);
  MemoryManager<Sphere> sphere_manager(2);
  int dummy;
  Sphere* sphered = sphere_manager.get_current_group(&dummy);

  sphered[0] = {{0.0f, 2.0f, 0.0f}, 1.5f, 1};
  sphered[1] = {{-3.0f, 5.0f, -2.0f}, 1.5f, 1};
  //sphered[1] = {{-2.0f, 5.0f, 0.0f}, 1.5f, 1};
  //sphered[2] = {{1.0f, -3.0f, 0.0f}, 1.5f, 0};

  /*for(int i = 2; i < 80; i++){
    sphered[i] = {vec3_factory((float)(i/20)*2.0f-4.0f, -0.5f, -8.0f-2.0f*(float)(i%20)), 1.0f, 4};
  }*/

  render_engine.set_sphere_memory(&sphere_manager);

  // Material Mem
  MaterialMem material_mem(6);
  material_mem.set_diffuse(0, {0.2f, 1.0f, 0.2f});
  material_mem.set_metal(1, {0.90f, 0.95f, 0.90f}, 0.000f);
  material_mem.set_diffuse(2, {1.0f, 1.0f, 1.0f});
  material_mem.set_diffuse(3, {0.8f, 0.8f, 1.0f});
  material_mem.set_diffuse(4, {0.8f, 0.4f, 0.2f});
  material_mem.set_diffuse(5, {0.9f, 0.1f, 0.1f});
  render_engine.set_material_memory(&material_mem);

  TriangleMemoryManager** tri_mem_managers = nullptr;
  size_t tri_mesh_count = 0;
  create_memory_manager("assets/objs/sewer_scene_detail.obj", 
                        &tri_mem_managers, tri_mesh_count, 3);

  render_engine.set_triangle_memory(tri_mem_managers, tri_mesh_count);

  LightMem light_memory;
  light_memory.pre_allocate_point_lights(3);
  Vec3 light_point = {-6.0f, 20.0f, -20.0f};
  Vec3 light_dir = vec3_get_norm(vec3_sub({2.0, 0.0, 2.0}, light_point));
  light_memory.assign_point_lights(0, {light_point,
                                       {1.0f, 0.1f, 0.1f},
                                       light_dir,
                                       0.6f, 20});
  Vec3 light_point2 = {-6.0f, 20.0f, 12.0f};
  Vec3 light_dir2 = vec3_get_norm(vec3_sub({2.0, 0.0, -2.0}, light_point2));
  light_memory.assign_point_lights(1, {light_point2,
                                       {0.1f, 0.1f, 1.0f},
                                       light_dir2,
                                       0.5f, 25});
  Vec3 light_point3 = {9.0f, 15.0f, -6.0f};
  Vec3 light_dir3 = vec3_get_norm(vec3_sub({0.0, 0.0, 0.0}, light_point3));
  light_memory.assign_point_lights(2, {light_point3,
                                       {0.1f, 1.0f, 0.1f},
                                       light_dir3,
                                       0.5f, 25});

  render_engine.set_light_memory(&light_memory);

  render_engine.render();
  render_engine.save_image(string("imgs/")+config.filename);

  return 0;
}



/*Sphere* in_one_weekend_spheres(){

  int sphere_count = 22*22+3+1;
  Sphere* sphere = (Sphere*)malloc(sizeof(Sphere)*sphere_count);


  int count = 0;
  Material metal = {{1.0f, 1.0f, 1.0f}, 1.0f, 0.0f, 0.0f};
  Material glass = {{1.0f, 1.0f, 1.0f}, 0.0f, 1.0f, 0.0f};
  for(int a = -11; a < 11; a++){
    for(int b = -11; b < 11; b++){

      Material diffuse = {{(float)drand48(), (float)float(drand48()), (float)drand48()}, 0.0f, 0.0f, 1.0f};
      metal.fuzz = (float)drand48()/2.0f;

      Material touse;
      float p = (float)drand48();
      if(p < 0.8){
        touse = diffuse;
      }
      else if(p < .95f){
        touse = metal;
      } 
      else{
        touse = glass;
      }

      Vec3 center = {a+0.9f*(float)drand48(), -0.8f, b+0.9f*(float)drand48()};
      init_sphere(sphere[count++], center, touse, 0.2f);

    }
  }

  init_sphere(sphere[count++], {0.0f, 0.0f, 0.0f}, {{0.5f, 0.6f, 1.0f},
      0.0f, 0.0f, 1.0f}, 1.0f);
  init_sphere(sphere[count++], {-4.0f, 0.0f, 0.0f}, {{1.0f, 1.0f, 1.0f},
      0.0f, 1.0f, 0.0f}, 1.0f);
  init_sphere(sphere[count++], {4.0f, 0.0f, 0.0f}, {{1.0f, 1.0f, 1.0f},
      1.0f, 0.0f, 0.0f}, 1.0f);
  init_sphere(sphere[count++], {0.0f, -601.0f, 0.0f}, 
      {{0.5f, 1.0f, 0.5f}, 0.0f, 0.0f, 1.0f}, 600.0f);
  printf("%d\n", count);
  return sphere;
}*/

