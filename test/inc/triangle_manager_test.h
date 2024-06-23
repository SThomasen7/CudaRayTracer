#ifndef _TRIANGLE_MANAGER_TEST_H_
#define _TRIANGLE_MANAGER_TEST_H_

#include "test.h"

// Test class
class TriangleManagerTestSuite : public TestSuite{

public:

  TriangleManagerTestSuite() : TestSuite("triangle_manager")
    { }

  // Test runner
  void run();

};

#endif
