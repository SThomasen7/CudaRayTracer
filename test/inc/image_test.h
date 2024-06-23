#ifndef _IMAGE_TEST_H_
#define _IMAGE_TEST_H_ 1

#include "test.h"

// Test class
class ImageTestSuite : public TestSuite{

public:

  ImageTestSuite() : TestSuite("image")
    { }

  // Test headers
  void run();

};


#endif
