#include "test.h"
#include "vec3_test.h"
#include "image_test.h"
#include "memory_manager_test.h"
#include "parser_test.h"
#include "triangle_manager_test.h"

int main(void){
  
  Vec3TestSuite v3_suite;
  ImageTestSuite img_suite;
  MemoryManagerTestSuite mem_suite;
  ParserTestSuite parse_suite;
  TriangleManagerTestSuite tri_suite;

  v3_suite.run();
  img_suite.run();
  mem_suite.run();
  parse_suite.run();
  tri_suite.run();

  return 0;
}
