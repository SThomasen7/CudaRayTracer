#include "parser_test.h"
#include "parser.h"

// Test case headers
bool parse_vec3(ostream& outstream, ostream& errstream);

// Run function to call the test cases.
void ParserTestSuite::run(){

    eval_test("parse_vec3", &parse_vec3);
    
    show_results();
}

// Test definitions
bool parse_vec3(ostream& outstream, ostream& errstream){
  bool result = true;
  
  Vec3 correct = vec3_factory(0.0f, 1.0f, 2.0f);
  Vec3 out = comma_sep_vec("0.0,1.0,2.0");

  assert_eq_v3(correct, out);

  out = comma_sep_vec("0,1,2");
 
  assert_eq_v3(correct, out);

  correct = vec3_factory(0.145f, 1.86f, 2.0002f);
  out = comma_sep_vec("0.145,1.86,2.0002");

  assert_eq_v3(correct, out);

  return result;
}
