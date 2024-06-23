#include "vec3_test.h"
#include "vec3.h"

bool init_vec3(ostream& outstream, ostream& errstream);
bool operators_vec3(ostream& outstream, ostream& errstream);
bool inplace_operators_vec3(ostream& outstream, ostream& errstream);
bool logical_operators_vec3(ostream& outstream, ostream& errstream);
bool linear_alg_vec3(ostream& outstream, ostream& errstream);

#define assert_equal_vec3(a,b) assert__eq_v3(outstream,errstream,a,b,__LINE__,result)
bool assert__eq_v3(ostream& outstream, ostream& errstream, Vec3& a, Vec3& b, int ln,
    bool& result);
// Test runners
void Vec3TestSuite::run(){

    eval_test("init_vec3", &init_vec3);
    eval_test("operators_vec3", &operators_vec3);
    eval_test("inplace_operators_vec3", &inplace_operators_vec3);
    eval_test("logical_operators_vec3", &logical_operators_vec3);
    eval_test("linear_alg_vec3", &linear_alg_vec3);
    
    show_results();
}

// Tests
// Tests simple copy and init operations of vec3.
// A little superfluous, but it'll be useful if we expand the vec3 class.
bool init_vec3(ostream& outstream, ostream& errstream){
  bool result = true;

  Vec3 a = {1.0f, 2.0f, 3.0f};
  Vec3 b = {4.0f, 5.0f, 6.0f};
  Vec3 c = b;

  assert_eq(c.x, b.x);
  assert_eq(c.y, b.y);
  assert_eq(c.z, b.z);

  assert_lt(a.x, b.x);
  assert_lt(a.y, b.y);
  assert_lt(a.z, b.z);

  assert_eq(a.x, 1.0f);
  assert_eq(a.y, 2.0f);
  assert_eq(a.z, 3.0f);

  return result;
}

bool operators_vec3(ostream& outstream, ostream& errstream){
  
  bool result = true;
  Vec3 a = {1.0f, 2.0f, 3.0f};
  Vec3 b = {4.0f, 5.0f, 6.0f};
  Vec3 c;
  Vec3 temp;

  // Vec add
  temp = {5.0f, 7.0f, 9.0f};
  c = vec3_add(a, b);
  assert_equal_vec3(c, temp);

  // Vec sub
  temp = {-3.0f, -3.0f, -3.0f};
  c = vec3_sub(a, b);
  assert_equal_vec3(c, temp);

  // vec mlt
  temp = {4.0f, 10.0f, 18.0f};
  c = vec3_mlt(a, b);
  assert_equal_vec3(c, temp);

  // vec div
  temp = {4.0f, 2.5f, 2.0f};
  c = vec3_div(b, a);
  assert_equal_vec3(c, temp);

  // vec add k
  float k;
  k = 8.3f;
  temp = {9.3f, 10.3f, 11.3f};
  c = vec3_add(a, k);
  assert_equal_vec3(c, temp);

  // vec sub k
  k = 0.5f;
  temp = {0.5f, 1.5f, 2.5f};
  c = vec3_sub(a, k);
  assert_equal_vec3(c, temp);

  // vec mlt k
  k = 2.0f;
  temp = {2.0f, 4.0f, 6.0f};
  c = vec3_mlt(a, k);
  assert_equal_vec3(c, temp);

  // vec div k
  k = 10.0f;
  temp = {0.1f, 0.2f, 0.3f};
  c = vec3_div(a, k);
  assert_equal_vec3(c, temp);
  
  return result;

}

bool inplace_operators_vec3(ostream& outstream, ostream& errstream){
  bool result = true;

  Vec3 a = {2.0f, 4.0f, 6.0f};
  Vec3 b = {5.0f, 3.0f, 1.0f};
  Vec3 c;
  Vec3 temp;

  // addequal test
  c = a;
  vec3_addequal(c, a);
  temp = {4.0f, 8.0f, 12.0f};
  assert_equal_vec3(temp, c);

  // subequal test
  c = b;
  vec3_subequal(c, a);
  temp = {3.0f, -1.0f, -5.0f};
  assert_equal_vec3(temp, c);

  // mltequal test
  c = b;
  vec3_mltequal(c, a);
  temp = {10.0f, 12.0f, 6.0f};
  assert_equal_vec3(temp, c);
  c = a;
  vec3_mltequal(c, b);
  assert_equal_vec3(temp, c);

  // divequal test
  c = a;
  vec3_divequal(c, a);
  temp = {1.0f, 1.0f, 1.0f};
  assert_equal_vec3(temp, c);
  c = b;
  vec3_divequal(c, a);
  temp = {2.5f, 0.75f, 0.16666666667f};
  assert_equal_vec3(temp, c);


  // Constant tests ***

  // addequal k test
  float k = 2.0f;
  c = a;
  vec3_addequal(c, k);
  temp = {4.0f, 6.0f, 8.0f};
  assert_equal_vec3(temp, c);

  // subequal k test
  c = a;
  vec3_subequal(c, k);
  temp = {0.0f, 2.0f, 4.0f};
  assert_equal_vec3(temp, c);

  // mltequal k test
  c = a;
  vec3_mltequal(c, k);
  temp = {4.0f, 8.0f, 12.0f};
  assert_equal_vec3(temp, c);

  // divequal k test
  c = a;
  vec3_divequal(c, k);
  temp = {1.0f, 2.0f, 3.0f};
  assert_equal_vec3(temp, c);

  return result;
}

bool logical_operators_vec3(ostream& outstream, ostream& errstream){
  bool result = true;

  Vec3 a = {0.0f, 2.0f, 4.0f};
  Vec3 b = {0.0f, 2.0f, 4.0f};

  assert_true(vec3_equals(a, b));
  assert_false(vec3_nequals(a, b));

  b.y = 0.0f;
  assert_false(vec3_equals(a, b));
  assert_true(vec3_nequals(a, b));

  return result;
}

bool linear_alg_vec3(ostream& outstream, ostream& errstream){
  bool result = true;

  Vec3 a = {2.0f, 4.0f, 6.0f};
  Vec3 b = {5.0f, 3.0f, 1.0f};

  // vec3_dot test
  assert_eq(vec3_dot(a, b), vec3_dot(b, a));
  assert_eq(vec3_dot(a, a), vec3_sqrd_len(a));
  assert_eq(vec3_dot(a, b), 28.0f);

  // vec3_cross test
  b = {1.0f, 3.0f, 5.0f};
  Vec3 temp = {-2.0f, 4.0f, -2.0f};

  Vec3 c = vec3_cross(b, a);
  assert_equal_vec3(temp, c);

  // vec3_len tests
  assert_eq(vec3_sqrd_len(a), 56.0f);
  assert_eq(vec3_len(a), sqrt(56.0f));
  
  temp = {8.0f, 0.0, 6.0f};
  b = temp;
  assert_eq(vec3_len(b), 10.0f);
  c = vec3_get_norm(b);
  Vec3 temp2 = {0.8f, 0.0f, 0.6f};
  assert_equal_vec3(temp2, c);
  assert_equal_vec3(temp, b);
  vec3_norm(b);
  assert_equal_vec3(temp2, b);

  return result;
}


// Test util
bool assert__eq_v3(ostream& outstream, ostream& errstream, Vec3& a, Vec3& b, int ln,
    bool &result){
  bool p = assert__eq_ln(outstream, errstream, a.x, b.x, ln, result);
  bool q = assert__eq_ln(outstream, errstream, a.y, b.y, ln, result);
  bool r =assert__eq_ln(outstream, errstream, a.z, b.z, ln, result);
  return p && q && r;
}
