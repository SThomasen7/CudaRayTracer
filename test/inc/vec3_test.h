#ifndef _VEC3_TEST_H_
#define _VEC3_TEST_H_ 1

#include "test.h"

// Test class
class Vec3TestSuite : public TestSuite{

public:

  Vec3TestSuite() : TestSuite("vec3")
    { }

  // Test headers
  void run();

};
#endif
