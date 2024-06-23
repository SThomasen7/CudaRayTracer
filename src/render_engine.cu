#include "render_engine.h"
#include "thread_mem.h"
#include "hit.h"
#include "ray.h"
#include "vec3.h"
#include <iostream>
#include "triangle.h"

using std::cout;
using std::endl;

//__constant__ float CONST_SHAPE_MEM[MAX_FLOATS_IN_CONST_MEM];

__global__ void look_at_mat_mem(float* dmat_mem, int num_mats);
__global__ void reset_thread_mem(int thread_count, ThreadMem* threadMem, int depth);
__global__ void hit_spheres(int thread_count, ThreadMem* threadMem, 
                            Sphere* sphere_dmem, int sphere_count, int depth);
__global__ void hit_spheres_shadow(int thread_count, ThreadMem* threadMem,
    Sphere* sphere_dmem, int sphere_count, int depth, Vec3 light_pos);
__global__ void hit_triangles(int thread_count, ThreadMem* threadMem, float* tri_dmem,
                            float* tri_dbsphere, int triangle_count, int depth, 
                            float material_index);
__global__ void hit_triangles_shadow(int thread_count, ThreadMem* threadMem, float* tri_dmem,
                            float* tri_dbsphere, int triangle_count, int depth, 
                            float material_index, Vec3 light_pos);

__global__ void redirect_rays(int thread_count, ThreadMem* threadMem);
__global__ void color_cells(int thread_count, ThreadMem* threadMem, float* material_mem,
    size_t depth);
__global__ void illum_mem_prep(int thread_count, ThreadMem* d_mem, 
                  Vec3 light_point, Vec3 light_direction, Vec3 cam_pos,
                  float light_angle);
__global__ void illum_mem(int thread_group, ThreadMem* dmem, Vec3 color,
                            Vec3 direction, Vec3 campos, float intensity);
__global__ void final_illum(int thread_group, ThreadMem* dmem);
__D__ float dist(Vec3 &a, Vec3& b);

// Constructors
RenderEngine::RenderEngine(SystemConfig sconfig, MemSize mem_size){
  config = sconfig;
  this->mem_size = mem_size;

  camera = new Camera(config.width, config.height, config.ray_samples,
      config.camera.position, config.camera.lookat, config.camera.up,
      config.camera.vfov);

  img = new Image(config.width, config.height, config.ray_samples);
  triangle_mesh_c = 0;

  this->block = dim3(1024/2);
  thread_count = img->width()*img->height()*config.ray_samples;
  if(thread_count % 32 == 0){
   grid = dim3((thread_count)/block.x);
  }
  else{
   grid = dim3(((thread_count)/block.x)+1);
  }

  sphere_mem = nullptr;
  tri_mem = nullptr;
  light_mem = nullptr;
  material_mem = nullptr;
}

RenderEngine::~RenderEngine(){
  delete img;
  delete camera;
}

// Set memory managers
void RenderEngine::set_sphere_memory(MemoryManager<Sphere>* memory_manager){
  sphere_mem = memory_manager;
}

void RenderEngine::set_triangle_memory(TriangleMemoryManager** memory_managers, size_t count){
  tri_mem = memory_managers;
  triangle_mesh_c = count;
}

void RenderEngine::set_material_memory(MaterialMem* memory_manager){
  material_mem = memory_manager;
}

void RenderEngine::set_light_memory(LightMem* light_memv){
  light_mem = light_memv;
}

// Write image out
void RenderEngine::save_image(std::string filename){
  img->writePPMtofile(filename);
}

// Render process
void RenderEngine::render(){

  cudaDeviceReset();
  int dummy = 0;
  MemoryManager<ThreadMem> thread_manager(thread_count);
  ThreadMem* threadmem = &thread_manager.get_current_group(&dummy)[0];
  camera->init_threadmem(threadmem);
  //thread_manager.copy_to_mem_manager(threadmem);
  
  // Allocate device mem
  ThreadMem* d_mem = thread_manager.alloc_gpu_shapemem();

  // Call the kernel to shoot these rays.
  cout << "Running... " << endl;
  for(int i = 0; i < config.recursion_depth; i++){
    do{
      // Get thread mem group
      int thread_group = 0;
      thread_manager.get_current_group_gpu(d_mem, &thread_group);


      // Ready Thread Mem to calculate hits.
      reset_thread_mem<<<grid,block>>>(thread_group, d_mem, i+1);
      CHECK_CUDA_ERROR(cudaPeekAtLastError());
      CHECK_CUDA_ERROR(cudaDeviceSynchronize());
      cout << "\b\b\b\b" << (int)(((float)(i+1)/
                (float)config.recursion_depth)*100.0f) << '%';
      std::flush(cout);

      // Process Sphere hits.
      if(sphere_mem != nullptr){
        sphere_pass(d_mem, thread_group, i);
      }

      // Iterate over all triangle meshes
      if(tri_mem != nullptr){
        triangle_pass(d_mem, thread_group, i);
      }

      // Update ray positions and collect colors
      float* d_matmem = material_mem->get_material_gpu_mem();

      color_cells<<<grid,block>>>(thread_group, d_mem, d_matmem, i+1);
      CHECK_CUDA_ERROR(cudaPeekAtLastError());
      CHECK_CUDA_ERROR(cudaDeviceSynchronize());
      material_mem->free_material_gpu_mem(d_matmem);

      // Ilumination pass
      if(light_mem != nullptr && i+1 < config.light_depth){
        illumination_pass(d_mem, thread_group, i+1);
      }

      redirect_rays<<<grid,block>>>(thread_group, d_mem);
      CHECK_CUDA_ERROR(cudaPeekAtLastError());
      CHECK_CUDA_ERROR(cudaDeviceSynchronize());

      /*reset_thread_mem<<<grid,block>>>(thread_group, d_mem, i+2);
      CHECK_CUDA_ERROR(cudaPeekAtLastError());
      CHECK_CUDA_ERROR(cudaDeviceSynchronize());*/

      if(i == config.recursion_depth -1){
        final_illum<<<grid,block>>>(thread_group, d_mem);
        CHECK_CUDA_ERROR(cudaPeekAtLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
      }


      thread_manager.copy_from_gpu_to_group(d_mem, &thread_group);

    }while(thread_manager.increment_group_idx());
  }

  // Update image raster
  img->update_raster(threadmem);
  
  // Free allocated memory
  cudaFree(d_mem);
  cudaDeviceReset();
  cout << endl << "Done." << endl;
}

// Render phases
// Hit all of the spheres
void RenderEngine::sphere_pass(ThreadMem* d_mem, int thread_group, int depth){

  Sphere* sphere_dmem = sphere_mem->alloc_gpu_shapemem();
  do{
    int shape_count = 0;
    sphere_mem->get_current_group_gpu(sphere_dmem, &shape_count);
    hit_spheres<<<grid,block>>>(thread_group, d_mem, sphere_dmem, 
                                shape_count, depth+1);

    CHECK_CUDA_ERROR(cudaPeekAtLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
  }
  while(sphere_mem->increment_group_idx());
  cudaFree(sphere_dmem);
}

void RenderEngine::sphere_shadow_pass(ThreadMem* d_mem, int thread_group, int depth, 
    Vec3 &light_pos){

  Sphere* sphere_dmem = sphere_mem->alloc_gpu_shapemem();
  do{
    int shape_count = 0;
    sphere_mem->get_current_group_gpu(sphere_dmem, &shape_count);
    //cout << "running sphere shadow pass" << endl;
    hit_spheres_shadow<<<grid,block>>>(thread_group, d_mem, sphere_dmem, 
                                shape_count, depth+1, light_pos);

    CHECK_CUDA_ERROR(cudaPeekAtLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
  }
  while(sphere_mem->increment_group_idx());
  cudaFree(sphere_dmem);
}

// Hit all of the triangles
void RenderEngine::triangle_pass(ThreadMem* d_mem, int thread_group, int depth){
  for(size_t tri_mem_idx = 0; tri_mem_idx < triangle_mesh_c; tri_mem_idx++){
    TriangleMemoryManager* curr_tri_mem = tri_mem[tri_mem_idx];

    // Process triangle hits
    float* tri_dbsphere = curr_tri_mem->alloc_gpu_bsphere();
    do{
      int shape_count = 0;
      float* tri_dmem = curr_tri_mem->alloc_gpu_shapemem();
      curr_tri_mem->get_current_group_gpu(tri_dmem, &shape_count);

      hit_triangles<<<grid,block>>>(thread_group, d_mem, tri_dmem,
          tri_dbsphere, shape_count, depth+1, curr_tri_mem->material_index);
      CHECK_CUDA_ERROR(cudaPeekAtLastError());
      CHECK_CUDA_ERROR(cudaDeviceSynchronize());
      cudaFree(tri_dmem);
    }
    while(curr_tri_mem->increment_group_idx());
    cudaFree(tri_dbsphere);
  }
}

void RenderEngine::triangle_shadow_pass(ThreadMem* d_mem, int thread_group, int depth,
        Vec3 &light_pos){
  for(size_t tri_mem_idx = 0; tri_mem_idx < triangle_mesh_c; tri_mem_idx++){
    TriangleMemoryManager* curr_tri_mem = tri_mem[tri_mem_idx];

    // Process triangle hits
    float* tri_dbsphere = curr_tri_mem->alloc_gpu_bsphere();
    do{
      int shape_count = 0;
      float* tri_dmem = curr_tri_mem->alloc_gpu_shapemem();
      curr_tri_mem->get_current_group_gpu(tri_dmem, &shape_count);

      hit_triangles_shadow<<<grid,block>>>(thread_group, d_mem, tri_dmem,
          tri_dbsphere, shape_count, depth+1, curr_tri_mem->material_index,
          light_pos);
      CHECK_CUDA_ERROR(cudaPeekAtLastError());
      CHECK_CUDA_ERROR(cudaDeviceSynchronize());
      cudaFree(tri_dmem);
    }
    while(curr_tri_mem->increment_group_idx());
    cudaFree(tri_dbsphere);
  }
}

void RenderEngine::illumination_pass(ThreadMem* dmem, int thread_group, int depth){
  point_light_pass(dmem, thread_group, depth);
}

/* phases in illumination pass*/
void RenderEngine::point_light_pass(ThreadMem* dmem, int thread_group, int depth){

  // For every object we will have to do a shadow hit, we can update the
  // hit record for auxillary memory because it will be reset after this
  // Just don't update t
  // we'll set v to non zero if hit
  for(size_t i = 0; i < light_mem->pl_count; i++){
    pointLight& current_light = light_mem->point_lights[i];

    // Prep for light check, first we check if the surface is facing away from
    // light and then we check if the light is powered and the angle of direction
    // is within the lights range.
    Vec3 light_pos = sample_point_light(current_light);
    illum_mem_prep<<<grid, block>>>(thread_group, dmem, 
                    light_pos, current_light.direction,
                    camera->origin, current_light.angle);
    CHECK_CUDA_ERROR(cudaPeekAtLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());


    // Next we will check all objects to determine if the light is obscured
    // Process Sphere hits.
    if(sphere_mem != nullptr){
      sphere_shadow_pass(dmem, thread_group, depth, light_pos);
    }

    // Iterate over all triangle meshes
    if(tri_mem != nullptr){
      triangle_shadow_pass(dmem, thread_group, depth, light_pos);
    }

    // Now we illuminate based on whether or not it is obstructed
    illum_mem<<<grid, block>>>(thread_group, dmem, current_light.color,
                            current_light.direction, camera->origin,
                            current_light.intensity);
    CHECK_CUDA_ERROR(cudaPeekAtLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  }
}

// GPU KERNELS ----------------------------------------------------------------
// Check all the spheres to see if there is a hit.
__global__ void hit_spheres(int thread_count, ThreadMem* threadMem,
    Sphere* sphere_dmem, int sphere_count, int depth){
  int tid = (threadIdx.x + blockDim.x*blockIdx.x);
  if(tid < 0 || tid >= thread_count){
    return;
  }

  ThreadMem* cell = &threadMem[tid];

  if(cell->hit.t < -0.001f){
    return;
  }

  for(int i = 0; i < sphere_count; i++){
    //Sphere sphere = get_sphere(i, CONST_SHAPE_MEM);
    sphere_hit(cell->ray, cell->hit, sphere_dmem[i]);
  }

}

__global__ void hit_spheres_shadow(int thread_count, ThreadMem* threadMem,
    Sphere* sphere_dmem, int sphere_count, int depth, Vec3 light_pos){
  int tid = (threadIdx.x + blockDim.x*blockIdx.x);
  if(tid < 0 || tid >= thread_count){
    return;
  }

  ThreadMem* cell = &threadMem[tid];

  if(cell->hit.t < -0.001f){
    return;
  }

  if(cell->hit.v > 0.0f){
    return;
  }

  curandState local_state;
  curand_init((int)clock64()+tid+1000*(int)(cell->hit.point.x+cell->hit.point.z+
        light_pos.y), 
      1, 0, &local_state);

  for(int i = 0; i < sphere_count; i++){
    //Sphere sphere = get_sphere(i, CONST_SHAPE_MEM);

    Ray ray = {cell->hit.point, vec3_get_norm(vec3_sub(light_pos, cell->hit.point))};

    float rx, ry, rz;
    rand_unit_vector(rx, ry, rz, local_state);
    ray.direction.x += rx * 0.1f;
    ray.direction.y += ry * 0.1f;
    ray.direction.z += rz * 0.1f;

    float tlim = dist(light_pos, cell->hit.point);
    if(sphere_shadow_hit(ray, cell->hit, sphere_dmem[i], tlim)){
      cell->hit.v = 1.0f;
      return;
    }
  }
}

// Check all the triangles to see if there is a hit.
__global__ void hit_triangles(int thread_count, ThreadMem* threadMem,
    float* tri_dmem, float* tri_dbsphere, int triangle_count, 
    int depth, float material_index){
  int tid = (threadIdx.x + blockDim.x*blockIdx.x);
  if(tid < 0 || tid >= thread_count){
    return;
  }

  ThreadMem* cell = &threadMem[tid];

  if(cell->hit.t < -0.001f){
    return;
  }

  // Check if we hit the bounding sphere
  Sphere bounding_sphere = {{tri_dbsphere[0], tri_dbsphere[1],
        tri_dbsphere[2]}, tri_dbsphere[3], 0};

  if(!sphere_shadow_hit(cell->ray, cell->hit, bounding_sphere, cell->hit.t)){
    // If it is within the sphere we still have to check
    if(dist(cell->ray.origin, bounding_sphere.center) > bounding_sphere.radius){
      return;
    }
  }


  for(int i = 0; i < triangle_count; i++){
    Triangle triangle = get_triangle(i, &tri_dmem[triangle_count*3],
                          &tri_dmem[0]);
    if(triangle_hit(cell->ray, cell->hit, triangle)){
      cell->hit.material_index = material_index;
    }
  }
}

__global__ void hit_triangles_shadow(int thread_count, ThreadMem* threadMem,
    float* tri_dmem, float* tri_dbsphere, int triangle_count, 
    int depth, float material_index, Vec3 light_pos){
  int tid = (threadIdx.x + blockDim.x*blockIdx.x);
  if(tid < 0 || tid >= thread_count){
    return;
  }

  ThreadMem* cell = &threadMem[tid];

  if(cell->hit.t < -0.001f){
    return;
  }

  if(cell->hit.v > 0.0f){
    return;
  }

  // No bounding check for this. :(
  curandState local_state;
  curand_init((int)clock64()+tid+1000*(int)(cell->hit.point.x), 1, 0, &local_state);

  for(int i = 0; i < triangle_count; i++){
    Triangle triangle = get_triangle(i, &tri_dmem[triangle_count*3],
                          &tri_dmem[0]);
    Ray ray = {cell->hit.point, vec3_sub(light_pos, cell->hit.point)};

    float rx, ry, rz;
    rand_unit_vector(rx, ry, rz, local_state);
    ray.direction.x += rx * 0.1f;
    ray.direction.y += ry * 0.1f;
    ray.direction.z += rz * 0.1f;

    float tlim = dist(light_pos, cell->hit.point);
    if(triangle_shadow_hit(ray, cell->hit, triangle, tlim)){
      cell->hit.v = 1.0f;
      break;
    }
  }
}


// Reset the t value for the valid threads.
__global__ void reset_thread_mem(int thread_count, ThreadMem* threadMem, int depth){

  int tid = (threadIdx.x + blockDim.x*blockIdx.x);
  if(tid < 0 || tid >= thread_count){
    return;
  }

  ThreadMem* cell = &threadMem[tid];
  if(cell->hit.t >= -0.001f){
    cell->hit.t = MAX_T;
  }

  cell->hit.point = {0.0f, 0.0f, 0.0f};
  cell->hit.normal = {0.0f, 0.0f, 0.0f};
  cell->hit.u = 0.0f;
  cell->hit.v = 0.0f;
  cell->hit.material_index = 0;
  cell->hit.front = true;
  // Just initilize, we don't over write this.

}

// Bounce rays and color the pixel
__global__ void redirect_rays(int thread_count, ThreadMem* threadMem){

  // Get thread id
  int tid = (threadIdx.x + blockDim.x*blockIdx.x);
  if(tid < 0 || tid >= thread_count){
    return;
  }

  // Access thread mem
  ThreadMem* cell = &threadMem[tid];
  
  if(cell->hit.t < -0.001f){
    return;
  }

  // If hit
  if(cell->hit.t < MAX_T){

    // Update the cell ray
    cell->ray.origin = cell->hit.point;

    // Send the ray
    if(cell->hit.t > 0.0f && cell->hit.t < 1.5f){
      reflect(cell->ray.direction, cell->hit.normal);
      cell->hit.prev_reflect = true;
    }
    else if(cell->hit.t >1.5f){
      refract(cell->ray.direction, cell->hit.normal, 1.5f, cell->hit.front);
      cell->hit.prev_reflect = true;
    }
    else{
      // Diffuse
      //FIXME come back to this? This seems wrong.
      cell->ray.direction = cell->hit.normal;
      cell->hit.prev_reflect = false;
    }

  }
}

__global__ void look_at_mat_mem(float* dmat_mem, int num_mats){
  int tid = (threadIdx.x + blockDim.x*blockIdx.x);

  if(tid >= num_mats){
    return;
  }

  Material hm = get_material(dmat_mem, tid);
  printf("%d, (%f,%f,%f), %f %f %f\n", tid, hm.color.x, hm.color.y,
      hm.color.z, hm.prob_reflect, hm.prob_refract, hm.fuzz);

}

// Colors the cells based on the material
__global__ void color_cells(int thread_count, ThreadMem* threadMem, float* material_mem,
    size_t depth){
  // Get thread id
  int tid = (threadIdx.x + blockDim.x*blockIdx.x);
  if(tid < 0 || tid >= thread_count){
    return;
  }

  // Accesss mem
  ThreadMem* cell = &threadMem[tid];

  if(cell->hit.t < -0.001f){
    return;
  }
  
  if(depth >= 50){
    cell->color = {0.0f, 0.0f, 0.0f};
    cell->hit.t = -1.0f;
    return;
  }

  // Get material
  Material hit_material = get_material(material_mem, cell->hit.material_index);

  Vec3 n = vec3_get_norm(cell->ray.direction);
  Vec3 out_color = {1.0f, 1.0f, 1.0f};

  // Set background color if not hit
  if(cell->hit.t == MAX_T){
    cell->hit.t = -1.0f;
    cell->hit.times_hit = 0;
    out_color = ray_color((n.y/2.0f)+0.5f);
    cell->color = vec3_mlt(cell->color, out_color);
    return;
  }

  // We've hit!
  if(!cell->hit.has_hit){
    out_color = hit_material.color;
    if(hit_material.fuzz == 1.0f){
      cell->hit.has_hit = true;
    }
  }

  // If diffuse
  if(hit_material.fuzz == 1.0f){
    float ratio = 1.0f;
    // If we've hit before we'll absorb some of the
    // energy
    if(cell->hit.times_hit != 0){
      ratio = pow(0.80f, cell->hit.times_hit);
    }
    cell->color = vec3_mlt(cell->color, ratio);
  }


  cell->color = vec3_mlt(cell->color, out_color);

  curandState local_state;
  curand_init((int)clock64()+tid+1000*(int)(cell->hit.point.x), 1, 0, &local_state);
  float p = curand_uniform(&local_state);

  // Determine how the ray should behave.
  //hit.t == 0          --> Diffuse
  //hit.t > 0 && < 1.5f --> Reflect
  // hit.t > 1.5f       --> Refract
  if(p < hit_material.prob_reflect){
    cell->hit.t = 1.0f;
  } else if(p < hit_material.prob_reflect + hit_material.prob_refract){
    cell->hit.t = 2.0f;
  } else{
    cell->hit.t = 0.0f;
    cell->hit.times_hit += 1;
  }

  // Add fuzz to the ray direction
  float rx, ry, rz;
  rx = cell->hit.point.x;
  ry = cell->hit.point.y;
  rz = cell->hit.point.z;
  
  rand_unit_vector(rx, ry, rz, local_state);
  cell->hit.normal.x += (rx * hit_material.fuzz);
  cell->hit.normal.y += (ry * hit_material.fuzz);
  cell->hit.normal.z += (rz * hit_material.fuzz);
}

__D__ float dist(Vec3& a, Vec3& b){
  return sqrt((a.x-b.x)*(a.x-b.x)+
              (a.y-b.y)*(a.y-b.y)+
              (a.z-b.z)*(a.z-b.z));
}

// This kernel will prep thread memory for the point light we are interested in
__global__ void illum_mem_prep(int thread_count, ThreadMem* d_mem, 
                  Vec3 light_point, Vec3 light_direction, Vec3 cam_pos,
                  float light_angle){

  // Get thread id
  int tid = (threadIdx.x + blockDim.x*blockIdx.x);
  if(tid < 0 || tid >= thread_count){
    return;
  }

  // Accesss mem
  ThreadMem* cell = &d_mem[tid];

  if(cell->hit.t < -0.001f || cell->hit.t == MAX_T){
    cell->hit.v = 1.0f;
    return;
  }

  // We only want to illuminate once we've hit our first diffuse surface.
  // But we'll still illuminate hitting a reflective object
  if(cell->hit.times_hit > 1){
    cell->hit.v = 1.0f;
    return;
  }
  
  //printf("Test\n");
  // v will be used to determine if we need to check for this light
  // (v > 0.0 if light is obstructed
  cell->hit.v = 0.0f;

  // Determine if the direction to this light is obstructed because of
  // the side we hit the object
  Vec3 lightn = vec3_get_norm(
                    vec3_sub(cell->hit.point, light_point)
                  );
  Vec3 minus_gaze = vec3_get_norm(vec3_sub(cam_pos, cell->hit.point));
  Vec3 normal = vec3_get_norm(cell->hit.normal);
  bool normalfaceslight = (vec3_dot(lightn, normal) 
                            < 0.0f);

  if(!normalfaceslight){
    cell->hit.v = 1.0f;
  }

  /*printf("to light(%f, %f, %f), \nminus_gaze(%f, %f, %f), \n hit_normal(%f, %f, %f), dot to_light, hit normal%f\n dot minus gaze, hit_normal %f\n, cell hit v%f \n\n", to_light.x, to_light.y, to_light.z, 
          minus_gaze.x,minus_gaze.y,minus_gaze.z,
          cell->hit.normal.x,cell->hit.normal.y,cell->hit.normal.z,
          vec3_dot(to_light, cell->hit.normal),
          vec3_dot(minus_gaze, cell->hit.normal), cell->hit.v);*/
  //printf("%f, %f\n", vec3_dot(minus_gaze, cell->hit.normal), cell->hit.v);

  // Determine if we are within the lights range (for directional light)
  // based on the angle of directional light
  if(light_angle >= 180){
    return;
  }

  if(light_angle <= 0){
    cell->hit.v = 1.0f;
  }
  
  // Translate the angle into something we can work with
  curandState local_state;
  curand_init((int)clock64()+tid+1000*(int)(cell->hit.point.x+cell->hit.point.z+
        normal.y), 1, 0, &local_state);
  float cosangle = cos(light_angle * (3.141592f/180.0f));
  cosangle += rand_float(local_state)*0.01f;
  float angle = vec3_dot(lightn, light_direction) / 
                  (vec3_len(lightn)*vec3_len(light_direction));

  //printf("%f, %f\n", cosangle, angle);
  //printf("%f, %f\n\n", light_angle, light_angle*(3.141592/180.0f));

  if (angle < cosangle){
    cell->hit.v = 1.0f;
  }

}
__global__ void illum_mem(int thread_count, ThreadMem* dmem, 
                            Vec3 lcolor, Vec3 ldirection, Vec3 cam_pos,
                            float intensity){

  // Get thread id
  int tid = (threadIdx.x + blockDim.x*blockIdx.x);
  if(tid < 0 || tid >= thread_count){
    return;
  }

  // Accesss mem
  ThreadMem* cell = &dmem[tid];

  if(cell->hit.t < -0.001f){
    return;
  }
  
  // Check if this point is obstructed
  if(cell->hit.v > 0.0f){
    return;
  }

  if(cell->hit.times_hit > 1){
    return;
  }

  // Illuminate the cell
  float specularexp = 128.0f;

  // Normal, light to point, and cam to point vectors
  Vec3 normal = vec3_get_norm(cell->hit.normal);
  Vec3 lightn = vec3_get_norm(vec3_sub(cell->hit.point, ldirection));
  //Vec3 gazedir = vec3_get_norm(cell->ray.direction);
  //Vec3 gazedir = vec3_get_norm(vec3_sub(cell->hit.point, cell->ray.origin));

  Vec3 gazedir = vec3_get_norm(vec3_sub(cell->hit.point, cam_pos));
  Vec3 N = vec3_mlt(normal, max(0.0f, vec3_dot(vec3_mlt(lightn, -1.0f), normal)));
  Vec3 R = vec3_add(vec3_mlt(N, 2.0f), lightn);

  float specular_factor = pow(max(0.0f, vec3_dot(R, vec3_mlt(gazedir, -1))), specularexp);
  float diffuse_factor = abs(vec3_dot(vec3_mlt(lightn, -1.0f), normal));
  //printf("%f, %f\n", diffuse_factor, intensity);

  float spec_mat = 0.9f;
  //float lightdist = dist(light, record.hit_p);
  //float atten = attenuation(12.0f, lightdist, scene->lineardecay[i]);

  Vec3 lightsdiffuse;
  Vec3 lightsspecular;

  // Get the light depending on linear or exponential decay

  //cell->color = vec3_add(cell->color, vec3_mlt(cell->color, vec3_add(lightsdiffuse, lightsspecular)));
  //printf("Ping\n");
  if(cell->hit.t == 0.0f){
    lightsdiffuse = vec3_mlt(lcolor, intensity * diffuse_factor);
    lightsspecular = vec3_mlt(lcolor, specular_factor * intensity * spec_mat);
    cell->light = vec3_add(cell->light, vec3_mlt(cell->color, lightsdiffuse));
    //cell->light = vec3_add(cell->light, vec3_mlt(cell->color, lightsspecular));
  }
  else{
    lightsdiffuse = vec3_mlt(lcolor, intensity * diffuse_factor * 0.3333f);
    lightsspecular = vec3_mlt(lcolor, specular_factor * intensity * spec_mat);
    cell->light = vec3_add(cell->light, vec3_mlt(cell->color, lightsdiffuse));
    //cell->light = vec3_add(cell->light, vec3_mlt(cell->color, lightsspecular));
  }
  //printf("(%f, %f, %f), (%f, %f, %f)\n", cell->color.x, cell->color.y, cell->color.z, 
          //lightsdiffuse.x,lightsdiffuse.y,lightsdiffuse.z);
}

__global__ void final_illum(int thread_count, ThreadMem* dmem){

  // Get thread id
  int tid = (threadIdx.x + blockDim.x*blockIdx.x);
  if(tid < 0 || tid >= thread_count){
    return;
  }

  // Accesss mem
  ThreadMem* cell = &dmem[tid];

  cell->color = vec3_add(cell->color, cell->light);
  //cell->color = cell->light;
  //cell->color = {1.0f, 1.0f, 1.0f};
}
