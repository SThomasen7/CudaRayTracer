#ifndef _MATERIAL_H_
#define _MATERIAL_H_

#include "vec3.h"

typedef struct Material{
  
  Vec3 color;
  float prob_reflect;
  float prob_refract;
  float fuzz;

} Material;

// The sum of these two should be less
// than or equal to 1. 1 minus these
// probs is for diffuse material.

#endif
