#ifndef _HIT_H_
#define _HIT_H_

#include "material.h"
#include "ray.h"

typedef struct HitRecord{
  Vec3 point;
  Vec3 normal;
  float t;
  float u, v;
  size_t material_index;
  int times_hit;
  bool front;
  bool prev_reflect;
  bool has_hit; // Has hit diffuse surface
} HitRecord;

__HD__ void init_hitrecord(HitRecord& record);

#endif
