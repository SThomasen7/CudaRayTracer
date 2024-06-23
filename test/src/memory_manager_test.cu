#include "memory_manager_test.h"
#include "memory_manager.h"
#include "test.h"

// Test case headers
bool init_memory_manager(ostream& outstream, ostream& errstream);
bool basic_group_test(ostream& outstream, ostream& errstream);
bool big_group_test(ostream& outstream, ostream& errstream);

// Some memory objects to test.
typedef struct MemObj1{
  float a;
  float b;
  float c;
} MemObj1;

typedef struct MemObj2{
  MemObj1 obj1a;
  MemObj1 obj1b;
  float d;
} MemObj2;

// Test runner
void MemoryManagerTestSuite::run(){

  eval_test("init_memory_manager", &init_memory_manager);
  eval_test("basic_group_test", &basic_group_test);
  eval_test("big_group_test", &big_group_test);

  show_results();
}


bool init_memory_manager(ostream& outstream, ostream& errstream){
  bool result = true;

  MemoryManager<MemObj2> testmanager = MemoryManager<MemObj2>(1200, 8192);

  assert_eq(sizeof(MemObj2), 28);
  assert_eq(testmanager.shapes_per_group(), 292);
  assert_eq(testmanager.num_groups(), 5);
  assert_eq(testmanager.current_group_id(), 0);
  assert_true(testmanager.increment_group_idx());
  assert_eq(testmanager.current_group_id(), 1);
  assert_true(testmanager.increment_group_idx());
  assert_true(testmanager.increment_group_idx());
  assert_true(testmanager.increment_group_idx());
  assert_false(testmanager.increment_group_idx());
  assert_eq(testmanager.current_group_id(), 0);
  assert_true(testmanager.increment_group_idx());
  testmanager.rewind();
  assert_eq(testmanager.current_group_id(), 0);
  assert_true(testmanager.increment_group_idx());
  assert_eq(testmanager.current_group_id(), 1);

  return result;
}


bool basic_group_test(ostream& outstream, ostream& errstream){
  bool result = true;

  // There should be three groups, one partially full
  MemoryManager<MemObj2> testmanager = MemoryManager<MemObj2>(1022, 8192);

  
  // Check the initialization details.
  assert_eq(testmanager.num_groups(), 4);
  assert_eq(testmanager.shapes_per_group(), 292);
  assert_eq(testmanager.current_group_id(), 0);
  
  // We're going to fill the memory and then assert that the floats are all correct.
  int i = 0;
  do{
    
    int data_count = 0;
    MemObj2* data = testmanager.get_current_group(&data_count);
  
    if(i == 3){
      assert_eq(data_count, 146);
    }else{
      assert_eq(data_count, 292);
    }

    for(int j = 0; j < data_count; j++){
      data[j].obj1a.a = (i*292*7)+(j*7)+0;
      data[j].obj1a.b = (i*292*7)+(j*7)+1;
      data[j].obj1a.c = (i*292*7)+(j*7)+2;
      data[j].obj1b.a = (i*292*7)+(j*7)+3;
      data[j].obj1b.b = (i*292*7)+(j*7)+4;
      data[j].obj1b.c = (i*292*7)+(j*7)+5;
      data[j].d =       (i*292*7)+(j*7)+6;
    }

    i++;
  }
  while(testmanager.increment_group_idx());


  int dummy;
  MemObj2* data = testmanager.get_current_group(&dummy);
  for(i = 0; i < 1022; i+=1){
    assert_eq((i*7)+0, (int)data[i].obj1a.a);
    assert_eq((i*7)+1, (int)data[i].obj1a.b);
    assert_eq((i*7)+2, (int)data[i].obj1a.c);
    assert_eq((i*7)+3, (int)data[i].obj1b.a);
    assert_eq((i*7)+4, (int)data[i].obj1b.b);
    assert_eq((i*7)+5, (int)data[i].obj1b.c);
    assert_eq((i*7)+6, (int)data[i].d);
  }

  return result;
}


bool big_group_test(ostream& outstream, ostream& errstream){
  bool result = true;
  // Set to true once implemented.

  MemoryManager<MemObj1> testmanager = MemoryManager<MemObj1>(4000, 12000);

  // Check the initialization details.
  assert_eq(testmanager.num_groups(), 4);
  assert_eq(testmanager.shapes_per_group(), 1000);
  assert_eq(testmanager.current_group_id(), 0);

  int i = 0;
  do{
    
    int data_count = 0;
    MemObj1* data = testmanager.get_current_group(&data_count);
  
    assert_eq(data_count, 1000);

    for(int j = 0; j < data_count; j++){
      data[j].a = (i*1000*3)+(j*3)+0;
      data[j].b = (i*1000*3)+(j*3)+1;
      data[j].c = (i*1000*3)+(j*3)+2;
    }

    i++;
  }
  while(testmanager.increment_group_idx());

  int dummy;
  MemObj1* data = testmanager.get_current_group(&dummy);
  for(i = 0; i < 4000; i+=1){
    assert_eq((i*3)+0, (int)data[i].a);
    assert_eq((i*3)+1, (int)data[i].b);
    assert_eq((i*3)+2, (int)data[i].c);
  }

  return result;
}


