#ifndef _CONFIG_H_
#define _CONFIG_H_

#include "vec3.h"
#include "string.h"

typedef struct CameraConfig{
  
  Vec3 position;
  Vec3 lookat;
  Vec3 up;
  float vfov;

} CameraConfig;

typedef struct SystemConfig{
  CameraConfig camera;
  std::string filename;

  /* Ray tracing details */
  int ray_samples; 
  int recursion_depth;
  int light_depth;

  /* Image details */
  int width;
  int height;


} SystemConfig;

void display_config(const SystemConfig &config);

#endif
