#ifndef _VEC3_H_
#define _VEC3_H_

#include "consts.h"
#include "ray_cuda_headers.h"

typedef struct Vec3{
  float x, y, z;
} Vec3;

__HD__ Vec3 vec3_factory(float x);
__HD__ Vec3 vec3_factory(float x, float y, float z);

// Operators
__HD__ Vec3 vec3_add(Vec3 a, Vec3 b);
__HD__ Vec3 vec3_sub(Vec3 a, Vec3 b);
__HD__ Vec3 vec3_mlt(Vec3 a, Vec3 b);
__HD__ Vec3 vec3_div(Vec3 a, Vec3 b);
__HD__ Vec3 vec3_add(Vec3 a, float k);
__HD__ Vec3 vec3_sub(Vec3 a, float k);
__HD__ Vec3 vec3_mlt(Vec3 a, float k);
__HD__ Vec3 vec3_div(Vec3 a, float k);


// Inplace operators
__HD__ void vec3_addequal(Vec3 &a, Vec3 &b);
__HD__ void vec3_subequal(Vec3 &a, Vec3 &b);
__HD__ void vec3_mltequal(Vec3 &a, Vec3 &b);
__HD__ void vec3_divequal(Vec3 &a, Vec3 &b);
__HD__ void vec3_addequal(Vec3 &a, float k);
__HD__ void vec3_subequal(Vec3 &a, float k);
__HD__ void vec3_mltequal(Vec3 &a, float k);
__HD__ void vec3_divequal(Vec3 &a, float k);

// Linear algebra operators
__HD__ float vec3_dot(Vec3 a, Vec3 b);
__HD__ Vec3 vec3_cross(Vec3 a, Vec3 b);
__HD__ float vec3_sqrd_len(Vec3 a);
__HD__ float vec3_len(Vec3 a);
__HD__ void vec3_norm(Vec3 &a);
__HD__ Vec3 vec3_get_norm(Vec3 a);

// Boolean
__HD__ bool vec3_equals(Vec3 &a, Vec3 &b);
__HD__ bool vec3_nequals(Vec3 &a, Vec3 &b);

__H__ void vec3_print(Vec3 &p);
#endif
