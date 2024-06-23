#include "camera.h"

float degrees_to_rad(float degrees){
  return (degrees * 3.14159265359f)/ 180.0f;
}

__H__ Camera::Camera(int width, int height, int samples, Vec3 origin,
  Vec3 look, Vec3 up, float vfov){

  image_height = height;
  image_width = width;
  this->samples = samples;

  float theta = degrees_to_rad(vfov);
  float h = tan(theta/2.0f);

  viewport_height = 2.0f * h;
  viewport_width = (width / height) * viewport_height;

  Vec3 w = vec3_get_norm(vec3_sub(origin, look));
  Vec3 u = vec3_get_norm(vec3_cross(up, w));
  Vec3 v = vec3_cross(w, u);

  this->origin = origin;
  horizontal = vec3_mlt(u, viewport_width);
  vertical = vec3_mlt(v, viewport_height);
  lower_left_corner = vec3_sub(origin, vec3_div(horizontal, 2.0f));
  lower_left_corner = vec3_sub(lower_left_corner, vec3_div(vertical, 2.0f));
  lower_left_corner = vec3_sub(lower_left_corner, w);

  /*printf("Origin: ");
  vec3_print(origin);
  printf(" Look: ");
  vec3_print(look);
  printf(" up: ");
  vec3_print(up);
  printf("\n");
  
  printf("w: ");
  vec3_print(w);
  printf(" u: ");
  vec3_print(u);
  printf(" v: ");
  vec3_print(v);
  printf("\n");
  
  printf("horizontal: ");
  vec3_print(horizontal);
  printf(" vertical: ");
  vec3_print(vertical);
  printf(" lower_left_corner: ");
  vec3_print(lower_left_corner);
  printf("\n");*/
}

__H__ Ray Camera::get_ray(float u, float v){
  Ray ray;
  ray.origin = origin;

  ray.direction = vec3_add(lower_left_corner, vec3_mlt(horizontal, u));
  ray.direction = vec3_add(ray.direction, vec3_mlt(vertical, v));
  ray.direction = vec3_sub(ray.direction, origin);

  return ray;
}

__H__ ThreadMem* Camera::init_threadmem(ThreadMem* h_mem){
  //int count = image_height*image_width*samples;

  // Set the seed
  srand48(SEED);

  // Populate the memory
  for(int x = 0; x < image_width; x++){
    for(int y = 0; y < image_height; y++){
      for(int s = 0; s < samples; s++){
        ThreadMem* thread = &h_mem[((x+y*image_width)*samples)+s];

        //float u = (float(x)) / (image_width-1);
        //float v = (float(y)) / (image_height-1);
        float u = (float(x)+drand48()-0.5f) / (image_width-1);
        float v = (float(image_height-1-y)+drand48()-0.5f) / (image_height-1);
        thread->ray = get_ray(u, v);
        thread->color = {1.0f, 1.0f, 1.0f};
        //thread->t = 100000.0f;
        //thread->count = 0;
        thread->hit.prev_reflect = false;
        thread->hit.has_hit = false;
        thread->hit.times_hit = 0;
        thread->light = {0.0f, 0.0f, 0.0f};
      }
    }
  }

  // Allocate and copy to the device.
  //ThreadMem* d_mem;
  //CHECK_CUDA_ERROR(cudaMalloc((void **)&d_mem, sizeof(ThreadMem)*count));
  //CHECK_CUDA_ERROR(cudaMemcpy(d_mem, h_mem, sizeof(ThreadMem)*count,
        //cudaMemcpyHostToDevice));

  //free(h_mem);
  //return d_mem;
  return h_mem;
}

// Copies thread mem to host.
/*__H__ ThreadMem* Camera::move_mem_to_host(ThreadMem* d_mem){
  int count = image_height*image_width*samples;
  ThreadMem* h_mem = (ThreadMem*)malloc(sizeof(ThreadMem)*count);
  CHECK_CUDA_ERROR(cudaMemcpy(h_mem, d_mem, sizeof(ThreadMem)*count,
        cudaMemcpyDeviceToHost));
  return h_mem;
}*/

// Host and device thread mem, either can be null
/*__H__ void free_threadmem(ThreadMem* d_ptr, ThreadMem* h_ptr){

  if(d_ptr != NULL && d_ptr != nullptr){
    CHECK_CUDA_ERROR(cudaFree(d_ptr));
    d_ptr = nullptr;
  }

  if(h_ptr != NULL && h_ptr != nullptr){
    free(h_ptr);
    h_ptr = nullptr;
  }
}*/

