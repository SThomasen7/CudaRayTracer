#include "hit.h"

#define MAX_T 100000.0f

__D__ void init_hitrecord(HitRecord& record){
  record.t = MAX_T;
}
