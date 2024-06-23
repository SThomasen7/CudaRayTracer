#ifndef _PARSER_TEST_H_
#define _PARSER_TEST_H_ 1

#include "test.h"

// Test class
class ParserTestSuite : public TestSuite{

public:

  ParserTestSuite() : TestSuite("parser")
    { }

  // Test headers
  void run();

};


#endif
