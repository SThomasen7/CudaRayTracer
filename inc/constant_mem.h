#ifndef _CONSTANT_MEM_H_
#define _CONSTANT_MEM_H_ 1

#include "consts.h"
#include "ray_cuda_headers.h"
/* NOTE DEPRECATED */
/* Constant memory is limited, but we will use it for two purposes:
 *
 *  1. Storing Lights
 *  2. Storing Materials (NOT TEXTURES)
 *
 *  To do this, we will create a memory manager that will distribute
 *  property values in an array of constant mem, and then a struct
 *  that describes how the memory is distributed.
 *  We can then use the structs with the kernels to access constant memory.
 *  
 *  There will be a CONSTANT_LIGHT array of floats, and a CONSTANT_MATERIAL
 *  array of floats.
 *  The material is for determining colors and such, while the lights are
 *  for determining the lighting of the scene.
 *
 */

//extern __constant__ float LIGHT_MEM[LIGHT_MEM_LEN];
// For now we will put a limit on how many materials we can have.
// Lights will start the same however we can probably change that using
// a Memory Manager later.

// Global variables keeping track of light and material counts

typedef struct LIGHTS{
  size_t point;
  size_t sphere;
  size_t disc;
  size_t plane;
  //size_t torus;
  size_t square;
} LIGHTS;


inline LIGHTS _NUM_LIGHTS_ = {
  0,
  0,
  0,
  0,
  //0,
  0
};

#endif
