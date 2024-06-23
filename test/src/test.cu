#include "test.h"
#include <stdexcept>

TestSuite::TestSuite(string name){
  fail_count = 0;
  success_count = 0;
  TestName = name;
  directory = "results/"+name+"/";

  outstream = ofstream(directory+name+".out");
  if(!outstream.is_open()){
    mark_fail("init_test");
    throw std::runtime_error("Failed creating output file stream");
  }

  errstream = ofstream(directory+name+".err");
  if(!errstream.is_open()){
    mark_fail("init_test");
    throw std::runtime_error("Failed creating error file stream");
  }

  mark_success("init_test");
}


TestSuite::~TestSuite(){
  outstream.close();
  errstream.close();
}

void TestSuite::mark_fail(string name){
  string fname = directory+name+".fail";
  results.push_back(name+".fail");
  fail_count++;
  ofstream file(fname);
  if(!file.is_open()){
    throw std::runtime_error("Failed creating .fail");
  }
  file.close();
}

void TestSuite::mark_success(string name){
  string fname = directory+name+".success";
  results.push_back(name+".success");
  success_count++;
  ofstream file(fname);
  if(!file.is_open()){
    throw std::runtime_error("Failed creating .success");
  }
  file.close();
}

void TestSuite::show_results(){
  std::cout << "Test Suite: " << TestName << endl;
  std::cout << "**********************************" << endl;
  std::cout << "Successes: " << success_count << endl;
  std::cout << "Fails: " << fail_count << endl;
  std::cout << "***" << endl;
  for(size_t i = 0; i < results.size(); i++){
    std::cout << results[i] << endl;
  }
  std::cout << "**********************************" << endl;
}

void TestSuite::eval_test(string test_name, bool (*test)(ostream&, ostream&)){
  outstream << "RUNNING TEST: " << test_name << endl;
  outstream << "**********************************" << endl;
  errstream << "RUNNING TEST: " << test_name << endl;
  errstream << "**********************************" << endl;
  bool state = test(outstream, errstream);
  outstream << "**********************************" << endl;
  errstream << "**********************************" << endl;
  outstream << "COMPLETED: " << (state ? "PASS" : "FAIL") << endl;
  outstream << "**********************************" << endl;
  errstream << "COMPLETED: " << (state ? "PASS" : "FAIL") << endl;
  errstream << "**********************************" << endl;
  if(state){
    mark_success(test_name);
  }
  else{
    mark_fail(test_name);
  }
}



// Some logical operators
bool assert__true_ln(ostream& outstream, ostream& errstream, bool test, int ln,
    bool &result){
  bool predicate = test;
  outstream << __FUNCTION__ << "(" << ln << "): ";
  if(predicate){
    outstream << "TRUE " << endl;
  }
  else{
    errstream << __FUNCTION__ << "(" << ln << "): ";
    errstream << "FALSE " << endl;
    outstream << "FALSE " << endl;
  }
  result = result && predicate;
  return predicate;
}

bool assert__false_ln(ostream& outstream, ostream& errstream, bool test, int ln,
    bool &result){
  bool predicate = !test;
  outstream << __FUNCTION__ << "(" << ln << "): ";
  if(predicate){
    outstream << "FALSE " << endl;
  }
  else{
    errstream << __FUNCTION__ << "(" << ln << "): ";
    errstream << "TRUE " << endl;
    outstream << "TRUE " << endl;
  }
  result = result && predicate;
  return predicate;
}


// FLOAT ASSERTIONS ---------------------------------------------------------------------
bool assert__lt_ln(ostream& outstream, ostream& errstream, float a, float b, int ln,
    bool &result){
  bool predicate = (a < b);
  outstream << __FUNCTION__ << "(" << ln << "): ";
  if(predicate){
    outstream << "TRUE (" << a << " < " << b << ")" << endl;
  }
  else{
    errstream << __FUNCTION__ << "(" << ln << "): ";
    errstream << "FALSE (" << a << " < " << b << ")" << endl;
    outstream << "FALSE (" << a << " < " << b << ")" << endl;
  }
  result = result && predicate;
  return predicate;
}

bool assert__gt_ln(ostream& outstream, ostream& errstream, float a, float b, int ln,
    bool &result){
  bool predicate = (a > b);
  outstream << __FUNCTION__ << "(" << ln << "): ";
  if(predicate){
    outstream << "TRUE (" << a << " > " << b << ")" << endl;
  }
  else{
    errstream << __FUNCTION__ << "(" << ln << "): ";
    errstream << "FALSE (" << a << " > " << b << ")" << endl;
    outstream << "FALSE (" << a << " > " << b << ")" << endl;
  }
  result = result && predicate;
  return predicate;
}

bool assert__lte_ln(ostream& outstream, ostream& errstream, float a, float b, int ln,
    bool &result){
  bool predicate = (a <= b);
  outstream << __FUNCTION__ << "(" << ln << "): ";
  if(predicate){
    outstream << "TRUE (" << a << " <= " << b << ")" << endl;
  }
  else{
    errstream << __FUNCTION__ << "(" << ln << "): ";
    errstream << "FALSE (" << a << " <= " << b << ")" << endl;
    outstream << "FALSE (" << a << " <= " << b << ")" << endl;
  }
  result = result && predicate;
  return predicate;
}

bool assert__gte_ln(ostream& outstream, ostream& errstream, float a, float b, int ln,
    bool &result){
  bool predicate = (a >= b);
  outstream << __FUNCTION__ << "(" << ln << "): ";
  if(predicate){
    outstream << "TRUE (" << a << " >= " << b << ")" << endl;
  }
  else{
    errstream << __FUNCTION__ << "(" << ln << "): ";
    errstream << "FALSE (" << a << " <= " << b << ")" << endl;
    outstream << "FALSE (" << a << " <= " << b << ")" << endl;
  }
  result = result && predicate;
  return predicate;
}

bool assert__eq_ln(ostream& outstream, ostream& errstream, float a, float b, int ln,
    bool &result){
  bool predicate = (abs(a - b) < 0.00001f);
  outstream << __FUNCTION__ << "(" << ln << "): ";
  if(predicate){
    outstream << "TRUE (" << a << " == " << b << ")" << endl;
  }
  else{
    errstream << __FUNCTION__ << "(" << ln << "): ";
    errstream << "FALSE (" << a << " == " << b << ")" << endl;
    outstream << "FALSE (" << a << " == " << b << ")" << endl;
  }
  result = result && predicate;
  return predicate;
}

bool assert__neq_ln(ostream& outstream, ostream& errstream, float a, float b, int ln,
    bool &result){
  bool predicate = (abs(a - b) > 0.00001f);
  outstream << __FUNCTION__ << "(" << ln << "): ";
  if(predicate){
    outstream << "TRUE (" << a << " != " << b << ")" << endl;
  }
  else{
    errstream << __FUNCTION__ << "(" << ln << "): ";
    errstream << "FALSE (" << a << " != " << b << ")" << endl;
    outstream << "FALSE (" << a << " != " << b << ")" << endl;
  }
  result = result && predicate;
  return predicate;
}

// VEC3 ASSERTIONS ---------------------------------------------------------------------
bool assert__eq_ln_v3(ostream& outstream, ostream& errstream, Vec3 &a, Vec3 &b, int ln, bool& result){ 
  bool predicate = vec3_equals(a, b);
  outstream << __FUNCTION__ << "(" << ln << "): ";
  if(predicate){
    outstream << "TRUE (" << a.x << " " << a.y << " " << a.z << " ==  ";
    outstream << b.x << " " << b.y << " " << b.z << ")" << endl;
  }
  else{
    errstream << __FUNCTION__ << "(" << ln << "): ";
    outstream << "FALSE (" << a.x << " " << a.y << " " << a.z << " ==  ";
    outstream << b.x << " " << b.y << " " << b.z << ")" << endl;
    errstream << "FALSE (" << a.x << " " << a.y << " " << a.z << " ==  ";
    errstream << b.x << " " << b.y << " " << b.z << ")" << endl;
  }
  result = result && predicate;
  return predicate;
}

bool assert__neq_ln_v3(ostream& outstream, ostream& errstream, Vec3 &a, Vec3 &b, int ln, bool& result){ 
  bool predicate = vec3_nequals(a, b);
  outstream << __FUNCTION__ << "(" << ln << "): ";
  if(predicate){
    outstream << "FALSE (" << a.x << " " << a.y << " " << a.z << " ==  ";
    outstream << b.x << " " << b.y << " " << b.z << ")" << endl;
  }
  else{
    errstream << __FUNCTION__ << "(" << ln << "): ";
    outstream << "TRUE (" << a.x << " " << a.y << " " << a.z << " ==  ";
    outstream << b.x << " " << b.y << " " << b.z << ")" << endl;
    errstream << "TRUE (" << a.x << " " << a.y << " " << a.z << " ==  ";
    errstream << b.x << " " << b.y << " " << b.z << ")" << endl;
  }
  result = result && predicate;
  return predicate;
}

// INT ASSERTIONS ---------------------------------------------------------------------
bool assert__eq_ln(ostream& outstream, ostream& errstream, int a, int b, int ln,
    bool &result){
  bool predicate = a == b;
  outstream << __FUNCTION__ << "(" << ln << "): ";
  if(predicate){
    outstream << "TRUE (" << a << " == " << b << ")" << endl;
  }
  else{
    errstream << __FUNCTION__ << "(" << ln << "): ";
    errstream << "FALSE (" << a << " == " << b << ")" << endl;
    outstream << "FALSE (" << a << " == " << b << ")" << endl;
  }
  result = result && predicate;
  return predicate;
}

bool assert__neq_ln(ostream& outstream, ostream& errstream, int a, int b, int ln,
    bool &result){
  bool predicate = a != b;
  outstream << __FUNCTION__ << "(" << ln << "): ";
  if(predicate){
    outstream << "TRUE (" << a << " != " << b << ")" << endl;
  }
  else{
    errstream << __FUNCTION__ << "(" << ln << "): ";
    errstream << "FALSE (" << a << " != " << b << ")" << endl;
    outstream << "FALSE (" << a << " != " << b << ")" << endl;
  }
  result = result && predicate;
  return predicate;
}

bool assert__lt_ln(ostream& outstream, ostream& errstream, int a, int b, int ln,
    bool &result){
  bool predicate = (a < b);
  outstream << __FUNCTION__ << "(" << ln << "): ";
  if(predicate){
    outstream << "TRUE (" << a << " < " << b << ")" << endl;
  }
  else{
    errstream << __FUNCTION__ << "(" << ln << "): ";
    errstream << "FALSE (" << a << " < " << b << ")" << endl;
    outstream << "FALSE (" << a << " < " << b << ")" << endl;
  }
  result = result && predicate;
  return predicate;
}

bool assert__gt_ln(ostream& outstream, ostream& errstream, int a, int b, int ln,
    bool &result){
  bool predicate = (a > b);
  outstream << __FUNCTION__ << "(" << ln << "): ";
  if(predicate){
    outstream << "TRUE (" << a << " > " << b << ")" << endl;
  }
  else{
    errstream << __FUNCTION__ << "(" << ln << "): ";
    errstream << "FALSE (" << a << " > " << b << ")" << endl;
    outstream << "FALSE (" << a << " > " << b << ")" << endl;
  }
  result = result && predicate;
  return predicate;
}

bool assert__lte_ln(ostream& outstream, ostream& errstream, int a, int b, int ln,
    bool &result){
  bool predicate = (a <= b);
  outstream << __FUNCTION__ << "(" << ln << "): ";
  if(predicate){
    outstream << "TRUE (" << a << " <= " << b << ")" << endl;
  }
  else{
    errstream << __FUNCTION__ << "(" << ln << "): ";
    errstream << "FALSE (" << a << " <= " << b << ")" << endl;
    outstream << "FALSE (" << a << " <= " << b << ")" << endl;
  }
  result = result && predicate;
  return predicate;
}

bool assert__gte_ln(ostream& outstream, ostream& errstream, int a, int b, int ln,
    bool &result){
  bool predicate = (a >= b);
  outstream << __FUNCTION__ << "(" << ln << "): ";
  if(predicate){
    outstream << "TRUE (" << a << " >= " << b << ")" << endl;
  }
  else{
    errstream << __FUNCTION__ << "(" << ln << "): ";
    errstream << "FALSE (" << a << " <= " << b << ")" << endl;
    outstream << "FALSE (" << a << " <= " << b << ")" << endl;
  }
  result = result && predicate;
  return predicate;
}
