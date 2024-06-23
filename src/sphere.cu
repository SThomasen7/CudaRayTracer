#include "sphere.h"
#include <stdio.h>
#include <exception>
#include <iostream>
using std::cout;
using std::endl;

__D__ float get_u(Vec3 point);
__D__ float get_v(Vec3 point);

/*extern __constant__ float CONST_SHAPE_MEM[16150];

SphereMemoryManager::SphereMemoryManager(size_t num_shapes) : 
  MemoryManager(num_shapes, sizeof(Sphere)){ }

SphereMemoryManager::SphereMemoryManager(size_t num_shapes, size_t mem_size) : 
  MemoryManager(num_shapes, sizeof(Sphere), mem_size){ }

void SphereMemoryManager::set_sphere(int index, Vec3 center, float radius,
      int material_index){

  if(index < 0){
    throw std::out_of_range("Negative value passed to SphereMemoryManager::set_sphere!");
  }
  else if(index >= get_num_shapes()){
    throw std::out_of_range("SphereMemoryManager::set_sphere index out of range.");
  }

  rewind();
  int dummy = 0;
  float* shapes = get_current_group(&dummy);

  float* sphere = &shapes[index*(sizeof(Sphere)/sizeof(float))];
  sphere[0] = center.x;
  sphere[1] = center.y;
  sphere[2] = center.z;
  sphere[3] = radius;
  sphere[4] = (float)material_index;
}

void SphereMemoryManager::set_sphere(int index, Sphere sphere){

  if(index < 0){
    throw std::out_of_range("Negative value passed to SphereMemoryManager::set_sphere!");
  }
  else if(index >= get_num_shapes()){
    throw std::out_of_range("SphereMemoryManager::set_sphere index out of range.");
  }

  rewind();
  int dummy = 0;
  float* shapes = get_current_group(&dummy);

  float* spheres = &shapes[index*(sizeof(Sphere)/sizeof(float))];
  spheres[0] = sphere.center.x;
  spheres[1] = sphere.center.y;
  spheres[2] = sphere.center.z;
  spheres[3] = sphere.radius;
  spheres[4] = (float)sphere.material_index;

}

*/
// Device functions
__D__ Sphere get_sphere(int index, float* shape_mem){

  float* sphere_loc = &shape_mem[index * (sizeof(Sphere)/sizeof(float))];
  Sphere out;
  out.center = { sphere_loc[0], sphere_loc[1], sphere_loc[2] };
  out.radius = sphere_loc[3];
  out.material_index = (size_t)sphere_loc[4];

  return out;
}

// Sphere hit
__D__ bool sphere_hit(Ray& ray, HitRecord& hit, Sphere& sphere){
  Vec3 oc = vec3_sub(ray.origin, sphere.center);

  float a = vec3_dot(ray.direction, ray.direction);
  float half_b = vec3_dot(oc, ray.direction);
  float c = vec3_dot(oc, oc) - sphere.radius*sphere.radius;
  float discriminant = half_b*half_b - a*c;

  if(discriminant < 0.001f){
    return false;
  }

  float sqrtd = sqrt(discriminant);
  float t = (-half_b - sqrtd)/ a;

  if(t < 0.001f || t > hit.t){
    t = (-half_b + sqrtd) / a;
    if (t < 0.001f || hit.t < t){
      return false;
    }
  }

  hit.t = t;
  hit.point = vec3_add(ray.origin, vec3_mlt(ray.direction, t));
  hit.normal = vec3_div(vec3_sub(hit.point, sphere.center), sphere.radius);
  hit.front = vec3_dot(ray.direction, hit.normal) < 0.0f;
  hit.normal = hit.front ? hit.normal : vec3_mlt(hit.normal, -1.0f);

  // Texture coordinates.
  hit.u = get_u(hit.normal);
  hit.v = get_v(hit.normal);
  
  hit.material_index = sphere.material_index;

  return true;
}

__D__ bool sphere_shadow_hit(Ray& ray, HitRecord& hit, Sphere& sphere,
    float t_lim){
  Vec3 oc = vec3_sub(ray.origin, sphere.center);

  float a = vec3_dot(ray.direction, ray.direction);
  float half_b = vec3_dot(oc, ray.direction);
  float c = vec3_dot(oc, oc) - sphere.radius*sphere.radius;
  float discriminant = half_b*half_b - a*c;

  if(discriminant < 0.001f){
    return false;
  }

  float sqrtd = sqrt(discriminant);
  float t = (-half_b - sqrtd)/ a;

  if(t < 0.001f || t > t_lim){
    t = (-half_b + sqrtd) / a;
    if (t < 0.001f || t_lim < t){
      return false;
    }
  }

  return true;
}

// Functions to calculate texture coordinates
__D__ float get_u(Vec3 point){
  float val = point.y;
  val = max(point.y, -1.0f);
  val = min(point.y, 1.0f);
  return acos(val)/3.14159265359f;

}

__D__ float get_v(Vec3 point){
  float x = point.x;
  float z = point.z;

  if(x == 0.0f && z == 0.0f){
    return 0.0f;
  }

  float phi = atan(z/x);
  float PI = 3.14159265359f;

  if(abs(x) < 0.0001f && z > 0.0f){
      phi = PI / 2.0f;
  } 
  else if(abs(x) < 0.001f && z <= 0.0f){
    phi = -PI / 2.0f;
  }

  if(x < 0.0f && z >= 0.0f){
      phi += PI;
  } 
  else if(x < 0.0f && z < 0.0f){
    phi -= PI;
  }

  return (phi + PI) / (2.0f * PI);

}

// CONST SHAPE MEM
/*void SphereMemoryManager::get_current_group_gpu(float* dmem, int* shape_count){
  float* hmem = get_current_group(shape_count);

  // Copy hmem to const mem
  cout << CONST_SHAPE_MEM << endl;
  cout << hmem << endl;
  cout << *shape_count << endl;
  cout << *shape_count*sizeof(Sphere) << endl;
  std::flush(cout);
  CHECK_CUDA_ERROR(cudaMemcpyToSymbol(CONST_SHAPE_MEM, hmem, 
        (size_t)(*shape_count)*sizeof(Sphere)));
}*/
