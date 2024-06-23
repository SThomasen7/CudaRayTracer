#include "triangle_manager_test.h"
#include "triangle_memory_manager.h"
#include "test.h"

// Test case headers
bool load_no_norm(ostream &outstream, ostream& errstream);
bool load_norm(ostream &outstream, ostream& errstream);
bool group_no_norm(ostream &outstream, ostream& errstream);
bool group_norm(ostream &outstream, ostream& errstream);

void TriangleManagerTestSuite::run(){

  eval_test("load_no_norm", &load_no_norm);
  eval_test("load_norm", &load_norm);
  eval_test("group_no_norm", &group_no_norm);
  eval_test("group_norm", &group_norm);

  show_results();
}

// Test the init values, number of groups, and vertex size
bool load_no_norm(ostream &outstream, ostream& errstream){
  bool result = true;

  TriangleMemoryManager tmm(5, 5*3, false, 0, 26*sizeof(float));

  float verts[] = { 
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,

        11.0f, 12.0f, 13.0f,
        14.0f, 15.0f, 16.0f,
        17.0f, 18.0f, 19.0f,

        21.0f, 22.0f, 23.0f,
        24.0f, 25.0f, 26.0f,
        27.0f, 28.0f, 29.0f,

        31.0f, 32.0f, 33.0f,
        34.0f, 35.0f, 36.0f,
        37.0f, 38.0f, 39.0f,

        41.0f, 42.0f, 43.0f,
        44.0f, 45.0f, 46.0f,
        47.0f, 48.0f, 49.0f
  };
    
  float index[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,
        11.0f, 12.0f, 13.0f,
        14.0f, 15.0f, 16.0f
  };

  tmm.load(&verts[0], &index[0]);
  
  assert_eq(tmm.num_groups(), 3);
  assert_eq(tmm.get_vert_size(), 3);

  return result;
}

// Test the init values, number of groups, and vertex size
bool load_norm(ostream &outstream, ostream& errstream){
  bool result = true;

  TriangleMemoryManager tmm2(5, 5*5, true, 0, 26*sizeof(float));

  float vertsN[] = { 
        1.0f, 2.0f, 3.0f, 3.33f, 6.66f, 9.99f,
        4.0f, 5.0f, 6.0f, 3.33f, 6.66f, 9.99f,
        7.0f, 8.0f, 9.0f, 3.33f, 6.66f, 9.99f,

        11.0f, 12.0f, 13.0f, 3.33f, 6.66f, 9.99f,
        14.0f, 15.0f, 16.0f, 3.33f, 6.66f, 9.99f,
        17.0f, 18.0f, 19.0f, 3.33f, 6.66f, 9.99f,

        21.0f, 22.0f, 23.0f, 3.33f, 6.66f, 9.99f,
        24.0f, 25.0f, 26.0f, 3.33f, 6.66f, 9.99f,
        27.0f, 28.0f, 29.0f, 3.33f, 6.66f, 9.99f,

        31.0f, 32.0f, 33.0f, 3.33f, 6.66f, 9.99f,
        34.0f, 35.0f, 36.0f, 3.33f, 6.66f, 9.99f,
        37.0f, 38.0f, 39.0f, 3.33f, 6.66f, 9.99f,

        41.0f, 42.0f, 43.0f, 3.33f, 6.66f, 9.99f,
        44.0f, 45.0f, 46.0f, 3.33f, 6.66f, 9.99f,
        47.0f, 48.0f, 49.0f, 3.33f, 6.66f, 9.99f
  };
    
  float indexN[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,
        11.0f, 12.0f, 13.0f,
        14.0f, 15.0f, 16.0f
  };

  tmm2.load(&vertsN[0], &indexN[0]);
  
  assert_eq(tmm2.num_groups(), 5);
  assert_eq(tmm2.get_vert_size(), 6);

  return result;
}

// Test the proper loading and setting of triangles without norms
bool group_no_norm(ostream &outstream, ostream& errstream){
  bool result = true;

  TriangleMemoryManager tmm(5, 5*3, false, 0, 26*sizeof(float));

  float verts[] = { 
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,

        11.0f, 12.0f, 13.0f,
        14.0f, 15.0f, 16.0f,
        17.0f, 18.0f, 19.0f,

        21.0f, 22.0f, 23.0f,
        24.0f, 25.0f, 26.0f,
        27.0f, 28.0f, 29.0f,

        31.0f, 32.0f, 33.0f,
        34.0f, 35.0f, 36.0f,
        37.0f, 38.0f, 39.0f,

        41.0f, 42.0f, 43.0f,
        44.0f, 45.0f, 46.0f,
        47.0f, 48.0f, 49.0f
  };
    
  float index[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,
        11.0f, 12.0f, 13.0f,
        14.0f, 15.0f, 16.0f
  };

  tmm.load(&verts[0], &index[0]);
  

  result = false;
  return result;
}

// Test the proper loading and setting of triangles with norms
bool group_norm(ostream &outstream, ostream& errstream){
  bool result = true;

  TriangleMemoryManager tmm2(5, 5*5, true, 0, 26*sizeof(float));

  float vertsN[] = { 
        1.0f, 2.0f, 3.0f, 3.13f, 6.16f, 9.19f,
        4.0f, 5.0f, 6.0f, 3.23f, 6.26f, 9.29f,
        7.0f, 8.0f, 9.0f, 3.33f, 6.36f, 9.39f,

        11.0f, 12.0f, 13.0f, 3.43f, 6.46f, 9.49f,
        14.0f, 15.0f, 16.0f, 3.53f, 6.56f, 9.59f,
        17.0f, 18.0f, 19.0f, 3.63f, 6.66f, 9.69f,

        21.0f, 22.0f, 23.0f, 3.73f, 6.76f, 9.79f,
        24.0f, 25.0f, 26.0f, 3.83f, 6.86f, 9.89f,
        27.0f, 28.0f, 29.0f, 3.93f, 6.96f, 9.99f,

        31.0f, 32.0f, 33.0f, 3.31f, 6.61f, 9.91f,
        34.0f, 35.0f, 36.0f, 3.32f, 6.62f, 9.92f,
        37.0f, 38.0f, 39.0f, 3.33f, 6.63f, 9.93f,

        41.0f, 42.0f, 43.0f, 3.34f, 6.64f, 9.97f,
        44.0f, 45.0f, 46.0f, 3.35f, 6.65f, 9.98f,
        47.0f, 48.0f, 49.0f, 3.36f, 6.66f, 9.99f
  };
    
  float indexN[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,
        11.0f, 12.0f, 13.0f,
        14.0f, 15.0f, 16.0f
  };

  result = false;
  return result;
}
