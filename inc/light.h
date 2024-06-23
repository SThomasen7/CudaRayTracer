#ifndef _LIGHT_H_
#define _LIGHT_H_

#include "vec3.h"

// Light poperties
typedef LightProperties{
  
  Vec3 position;
  Vec3 intensity; // Color
  Vec3 focused;   // Normal direction pointing the light
  float angle;    // In degrees
  //bool powered;
  // NOTE: Turn the light off by setting angle to 0.

  // If angle > 330 focused is ignored and it is treated like
  // an infinite light
  
} LightProperties;


// Define various types of light
typedef struct PointLight{
  LightProperties props;
};

typedef struct SphereLight{
  LightProperties props;
  float radius;
};

typedef struct DiscLight{
  LightProperties props;
  float radius;
};

// Infinite plane
typedef struct PlaneLight{
  Vec3 normal;
  LightProperties props;
  float point;
};

/*typedef struct TorusLight{
  LightProperties props;
  float disc_radius;
  float tube_radius;
};*/

typedef struct SquareLight{
  /*
   *  v1 -------- v2
   *   |          |
   *   |          |
   *   |          |
   *   |          |
   *  v1 -------- v2
   */

  Vec3 v1, v2;
  Vec3 v3, v4;
  LightProperties props;
}

#endif
