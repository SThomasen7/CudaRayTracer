#ifndef _THREAD_MEM_H_
#define _THREAD_MEM_H_ 1

#include "ray.h"
#include "hit.h"

// t will be used if we need more than one
// kernel call to calculate all the objects
// in the scene.
// If the ray doesn't hit anything, we'll set
// t to 0.0f
typedef struct ThreadMem{
  Ray ray;
  Vec3 color;
  Vec3 light;
  HitRecord hit;
} ThreadMem;
// 23 Floats (out dated)

#endif
