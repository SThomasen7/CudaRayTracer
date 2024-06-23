#include "vec3.h"
#include <stdio.h>

__HD__ Vec3 vec3_factory(float x){
  Vec3 out = {x, x, x};
  return out;
}

__HD__ Vec3 vec3_factory(float x, float y, float z){
  Vec3 out = {x, y, z};
  return out;
}

// Operators
__HD__ Vec3 vec3_add(Vec3 a, Vec3 b){
  Vec3 out;
  out.x = a.x + b.x;
  out.y = a.y + b.y;
  out.z = a.z + b.z;
  return out;
}

__HD__ Vec3 vec3_sub(Vec3 a, Vec3 b){
  Vec3 out;
  out.x = a.x - b.x;
  out.y = a.y - b.y;
  out.z = a.z - b.z;
  return out;
}

__HD__ Vec3 vec3_mlt(Vec3 a, Vec3 b){
  Vec3 out;
  out.x = a.x * b.x;
  out.y = a.y * b.y;
  out.z = a.z * b.z;
  return out;
}

__HD__ Vec3 vec3_div(Vec3 a, Vec3 b){
  Vec3 out;
  out.x = a.x / b.x;
  out.y = a.y / b.y;
  out.z = a.z / b.z;
  return out;
}

__HD__ Vec3 vec3_add(Vec3 a, float k){
  Vec3 out;
  out.x = a.x + k;
  out.y = a.y + k;
  out.z = a.z + k;
  return out;
}

__HD__ Vec3 vec3_sub(Vec3 a, float k){
  Vec3 out;
  out.x = a.x - k;
  out.y = a.y - k;
  out.z = a.z - k;
  return out;
}

__HD__ Vec3 vec3_mlt(Vec3 a, float k){
  Vec3 out;
  out.x = a.x * k;
  out.y = a.y * k;
  out.z = a.z * k;
  return out;
}

__HD__ Vec3 vec3_div(Vec3 a, float k){
  Vec3 out;
  out.x = a.x / k;
  out.y = a.y / k;
  out.z = a.z / k;
  return out;
}



// Inplace operators
__HD__ void vec3_addequal(Vec3 &a, Vec3 &b){
  a = vec3_add(a, b);
}

__HD__ void vec3_subequal(Vec3 &a, Vec3 &b){
  a = vec3_sub(a, b);
}

__HD__ void vec3_mltequal(Vec3 &a, Vec3 &b){
  a = vec3_mlt(a, b);
}

__HD__ void vec3_divequal(Vec3 &a, Vec3 &b){
  a = vec3_div(a, b);
}

__HD__ void vec3_addequal(Vec3 &a, float k){
  a = vec3_add(a, k);
}

__HD__ void vec3_subequal(Vec3 &a, float k){
  a = vec3_sub(a, k);
}

__HD__ void vec3_mltequal(Vec3 &a, float k){
  a = vec3_mlt(a, k);
}

__HD__ void vec3_divequal(Vec3 &a, float k){
  a = vec3_div(a, k);
}


// Linear algebra operators
__HD__ float vec3_dot(Vec3 a, Vec3 b){
  return a.x*b.x + a.y*b.y + a.z*b.z;
}

__HD__ Vec3 vec3_cross(Vec3 a, Vec3 b){
  Vec3 out;
  out.x = a.y*b.z - a.z*b.y;
  out.y = a.z*b.x - a.x*b.z;
  out.z = a.x*b.y - a.y*b.x;
  return out;
}

__HD__ float vec3_sqrd_len(Vec3 a){
  return vec3_dot(a, a);
}

__HD__ float vec3_len(Vec3 a){
  return sqrt(vec3_dot(a, a));
}


__HD__ void vec3_norm(Vec3 &a){
  vec3_divequal(a, vec3_len(a));
}

__HD__ Vec3 vec3_get_norm(Vec3 a){
  return vec3_div(a, vec3_len(a));
}


// Boolean
__HD__ bool vec3_equals(Vec3 &a, Vec3 &b){
  return ((abs(a.x - b.x) < 0.0001) &&
          (abs(a.y - b.y) < 0.0001) &&
          (abs(a.z - b.z) < 0.0001));
}

__HD__ bool vec3_nequals(Vec3 &a, Vec3 &b){
  return !vec3_equals(a, b);
}

__H__ void vec3_print(Vec3 &p){
  printf("(%.4f, %.4f, %.4f)", p.x, p.y, p.z);
}
