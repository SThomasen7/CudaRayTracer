#ifndef _RAY_TRACING_H_
#define _RAY_TRACING_H_

#include "ray_cuda_headers.h"
#include "vec3.h"

__D__ Vec3 ray_color(float t);
__D__ float rand_float(curandState& state);
__D__ void rand_unit_vector(float &x, float &y, float &z, curandState &local_state);

__D__ void reflect(Vec3 &direction, Vec3 normal);
__D__ void refract(Vec3& direction, Vec3 normal, float index_refraction,
      bool front_face/*, float rand*/);
__D__ float reflectance(float cos_theta, float refraction_ratio);


#endif
