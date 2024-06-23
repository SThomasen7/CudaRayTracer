#ifndef _TRIANGLE_H_
#define _TRIANGLE_H_ 1

#include "ray_cuda_headers.h"
#include "material.h"
#include "vec3.h"
#include "memory_manager.h"
#include "hit.h"
#include "ray.h"
#include "lights.h"

typedef struct Vertex{
  Vec3 point;
} Vertex;

typedef struct VertexN{
  Vec3 point;
  Vec3 normal;
} VertexN;

// Unlike a sphere I expect we will pass the material index for a mesh
// to the kernel, i.e. we will probably only render one mesh at a time
// given how much memory triangles will need compared to spheres.
typedef struct Triangle{
  Vec3 v1, v2, v3;
  //float material_index;
} Triangle;

typedef struct TriangleN{
  Vec3 v1, v2, v3;
  Vec3 n1, n2, n3;
  //float material_index;
} TriangleN;

__D__ Triangle get_triangle(int index, float* vert_mem, float* index_buffer);
__D__ TriangleN get_triangleN(int index, float* vert_mem, float* index_buffer);
__D__ bool triangle_hit(Ray& ray, HitRecord& hit, Triangle& triangle);
__D__ bool triangle_shadow_hit(Ray& ray, HitRecord& hit, Triangle& triangle, float tlim);
__D__ bool triangle_hitN(Ray& ray, HitRecord& hit, TriangleN& triangle);

#endif
