#ifndef _CAMERA_H_
#define _CAMERA_H_ 1

#include "ray_cuda_headers.h"

#include "vec3.h"
#include "thread_mem.h"

#define SEED 230711

class Camera{
public:

  __H__ Camera(int width, int height, int samples, 
      Vec3 origin, Vec3 look, Vec3 up, float vfov);
  __H__ ~Camera(){ }
  

  __H__ Ray get_ray(float u, float v);

  __H__ ThreadMem* init_threadmem(ThreadMem* d_mem);
  //__H__ ThreadMem* move_mem_to_host(ThreadMem* d_mem);
  //__H__ void free_threadmem(ThreadMem* ptr);

  Vec3 origin;

private:

  Vec3 lower_left_corner;
  Vec3 horizontal;
  Vec3 vertical;

  float viewport_width;
  float viewport_height;
  float focal_length;

  int image_width;
  int image_height;
  int samples;


};

#endif
