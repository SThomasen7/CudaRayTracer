#ifndef _MEMORY_MANAGER_TEST_H_
#define _MEMORY_MANAGER_TEST_H_

#include "test.h"

// Test class
class MemoryManagerTestSuite : public TestSuite{

public:

  MemoryManagerTestSuite() : TestSuite("memory_manager")
    { }

  // Test runner
  void run();

};

#endif
