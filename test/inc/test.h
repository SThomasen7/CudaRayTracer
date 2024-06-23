#ifndef _TEST_H_
#define _TEST_H_

#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include "vec3.h"

using std::string;
using std::ofstream;
using std::ostream;

using std::endl;
using std::vector;


#define assert_true(a) assert__true_ln(outstream, errstream, a,__LINE__,result)
#define assert_false(a) assert__false_ln(outstream, errstream, a,__LINE__,result)

#define assert_lt(a,b) assert__lt_ln(outstream, errstream, a, b, __LINE__,result)
#define assert_gt(a,b) assert__gt_ln(outstream, errstream, a, b, __LINE__,result)
#define assert_lte(a,b) assert__lte_ln(outstream, errstream, a, b, __LINE__,result)
#define assert_gte(a,b) assert__gte_ln(outstream, errstream, a, b, __LINE__,result)
#define assert_eq(a,b) assert__eq_ln(outstream, errstream, a, b, __LINE__,result)
#define assert_ne(a,b) assert__ne_ln(outstream, errstream, a, b, __LINE__,result)
#define assert_eq_v3(a,b) assert__eq_ln_v3(outstream, errstream, a, b, __LINE__,result)
#define assert_ne_v3(a,b) assert__ne_ln_v3(outstream, errstream, a, b, __LINE__,result)

// Some logical operators
bool assert__true_ln(ostream& out, ostream& err, bool test, int ln, bool& result);
bool assert__false_ln(ostream& out, ostream& err, bool test, int ln, bool& result);

bool assert__lt_ln(ostream& out, ostream& err, float a, float b, int ln, bool& result);
bool assert__gt_ln(ostream& out, ostream& err, float a, float b, int ln, bool& result);
bool assert__lte_ln(ostream& out, ostream& err, float a, float b, int ln, bool& result);
bool assert__gte_ln(ostream& out, ostream& err, float a, float b, int ln, bool& result);
bool assert__eq_ln(ostream& out, ostream& err, float a, float b, int ln, bool& result);
bool assert__neq_ln(ostream& out, ostream& err, float a, float b, int ln, bool& result);
bool assert__eq_ln_v3(ostream& out, ostream&err, Vec3 &a, Vec3 &b, int ln, bool& result);
bool assert__neq_ln_v3(ostream& out, ostream&err, Vec3 &a, Vec3 &b, int ln, bool& result);

bool assert__eq_ln(ostream& out, ostream& err, int a, int b, int ln, bool& result);
bool assert__neq_ln(ostream& out, ostream& err, int a, int b, int ln, bool& result);
bool assert__lt_ln(ostream& out, ostream& err, int a, int b, int ln, bool& result);
bool assert__gt_ln(ostream& out, ostream& err, int a, int b, int ln, bool& result);
bool assert__lte_ln(ostream& out, ostream& err, int a, int b, int ln, bool& result);
bool assert__gte_ln(ostream& out, ostream& err, int a, int b, int ln, bool& result);

class TestSuite{

public:
  // Provide the test suite name.
  TestSuite(string name);
  ~TestSuite();

  virtual void run() = 0;

  void mark_fail(string name);
  void mark_success(string name);

  void show_results();

  void eval_test(string test_name, bool (*test)(ostream&, ostream&));
    
  ofstream outstream;
  ofstream errstream;


private:

  string TestName;
  string directory;

  vector<string> results;
  int fail_count;
  int success_count;

};

#endif
