#include "ray_tracing.h"

// Get the background color based on the ray
__D__ Vec3 ray_color(float t){
  //return vec3_add(vec3_mlt({1.0f, 1.0f, 1.0f}, (1.0f-t)), vec3_mlt({0.5f, 0.7f, 1.0f}, t));
  //return vec3_add(vec3_mlt({1.0f, 1.0f, 1.0f}, (1.0f-t)), vec3_mlt({0.5f, 0.7f, 1.0f}, t));
  //return vec3_add(vec3_mlt({.5f, .5f, .5f}, (1.0f-t)), vec3_mlt({0.25f, 0.25f, 0.5f}, t));
  return {.2, .2, .2};
}

__D__ float rand_float(curandState& state){
  float f = (curand_uniform(&state) - 0.5f)*2.0f;
  //printf("%f\n", f);
  return f;
}

__D__ void rand_unit_vector(float &x, float &y, float &z,
    curandState &local_state){
  float length = 0.0f;
  x = rand_float(local_state);
  y = rand_float(local_state);
  z = rand_float(local_state);
  length = sqrt(x*x+y*y+z*z);
  x /= length;
  y /= length;
  z /= length;
}


__D__ void reflect(Vec3 &direction, Vec3 normal){
  // Calculate the reflected ray unless it's grazing the edge of
  // the sphere
  if(abs(vec3_dot(direction, normal)) > 0.0001f){
    direction = vec3_sub(direction, 
        vec3_mlt(
          normal,
          vec3_dot(direction, normal) * 2.0f
        )
      );
  }
}

__D__ void refract(Vec3& direction, Vec3 normal, float index_refraction,
      bool front_face) {
  float refraction_ratio = front_face ? (1.0f/index_refraction) : index_refraction; 
  Vec3 ndirection = vec3_get_norm(direction);
  float cos_theta = min(vec3_dot(vec3_mlt(ndirection, -1.0), normal), 1.0);
  if(((refraction_ratio * sqrt(1.0f-cos_theta*cos_theta)) > 1.0f)/* ||
      reflectance(cos_theta, refraction_ratio) > reflectance_prob*/){
    reflect(ndirection, normal);
    return;
  }
  Vec3 r_out_perp = vec3_mlt(
      vec3_add(
        ndirection, 
        vec3_mlt(
          normal, 
          cos_theta
        )
      ), 
      refraction_ratio
    );
  Vec3 r_out_parallel = vec3_mlt(normal, -sqrt(abs(1.0 - vec3_dot(r_out_perp, r_out_perp))));
  direction = vec3_add(r_out_perp, r_out_parallel);
}

__D__ float reflectance(float cos_theta, float refraction_ratio){
  float r0 = (1.0f-refraction_ratio) / (1+refraction_ratio);
  r0 = r0*r0;
  return r0 + (1.0f-r0) * pow((1.0f-cos_theta), 5);
}
