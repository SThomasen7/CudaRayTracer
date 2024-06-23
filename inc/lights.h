#ifndef _LIGHT_H_
#define _LIGHT_H_
#include "ray_cuda_headers.h"
#include "consts.h"
#include "vec3.h"

/*typedef struct LIGHTS{
  size_t point;
} LIGHTS;

// Keep track of number of lights
inline LIGHTS _LIGHT_COUNTS_{
  0
};*/


// Types of lights
typedef struct pointLight{
  Vec3 pos;
  Vec3 color;
  Vec3 direction;
  float intensity;
  float angle; // 0 to 180
               // 0 when powered off
               // 180 to ignore direction
} pointLight;

class LightMem{
public:
  LightMem();
  ~LightMem();

  void pre_allocate_point_lights(size_t num_point);
  void assign_point_lights(size_t idx, pointLight light);

  inline bool has_point_lights(){ return pl_count > 0; }

  // Allocate and copy point light memory to GPU
  pointLight* get_gpu_mem_pointlight(); 

  pointLight* point_lights;
  size_t pl_count;
  
};

// Sample a point on lights
__HD__ Vec3 sample_point_light(pointLight& light);

#endif
