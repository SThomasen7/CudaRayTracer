#include "image_test.h"
#include "image.h"
#include <sstream>

using std::ostringstream;

// Test case headers
bool init_image(ostream& outstream, ostream& errstream);
bool clamp_image(ostream& outstream, ostream& errstream);
bool uv_image(ostream& outstream, ostream& errstream);
bool idxs_image(ostream& outstream, ostream& errstream);
bool write_image(ostream& outstream, ostream& errstream);
bool read_image(ostream& outstream, ostream& errstream);
//bool raster_update_image(ostream& outstream, ostream& errstream);
//bool raster_update_image2(ostream& outstream, ostream& errstream);

#define compare_imgs(a,b) compare_imgs_mac(a,b,outstream,errstream,result)
void compare_imgs_mac(Image& img, string& output, ostream& outstream, 
    ostream& errstream, bool &result);

// A test util

// Run function to call the test cases.
void ImageTestSuite::run(){

    eval_test("init_image", &init_image);
    eval_test("clamp_image", &clamp_image);
    eval_test("uv_image", &uv_image);
    eval_test("idxs_image", &idxs_image);
    eval_test("write_image", &write_image);
    eval_test("read_image", &read_image);
    //eval_test("raster_update_image", &raster_update_image);
    //eval_test("raster_update_image2", &raster_update_image2);
    
    show_results();
}

// Test definitions
// Tests the init function, width and height.
bool init_image(ostream& outstream, ostream& errstream){
  bool result = true;
  
  Image img(3, 4, 5);

  assert_eq(3, img.width());
  assert_eq(4, img.height());
  assert_eq(5, img.samples());

  return result;
}

// Test the clamp functions
bool clamp_image(ostream& outstream, ostream& errstream){
  bool result = true;

  // Test the tree clamp functions.
  // Float pointer
  float rgb[3] = {-1.0f, 2.0f, 1.1f};
  clamp(rgb);

  assert_eq(rgb[0], 0.0f);
  assert_eq(rgb[1], 1.0f);
  assert_eq(rgb[2], 1.0f);

  rgb[0] = 0.333f;
  rgb[1] = 0.25f;
  rgb[2] = 0.9f;
  clamp(rgb);
  assert_eq(rgb[0], 0.333f);
  assert_eq(rgb[1], 0.25f);
  assert_eq(rgb[2], 0.9f);

  // Vec3
  Vec3 test = {8.0f, -10.0f, 1.2f};
  clamp(test);

  assert_eq(test.x, 1.0f);
  assert_eq(test.y, 0.0f);
  assert_eq(test.z, 1.0f);

  test.x = 0.8f;
  test.y = 0.3f;
  test.z = 0.1f;

  assert_eq(test.x, 0.8f);
  assert_eq(test.y, 0.3f);
  assert_eq(test.z, 0.1f);


  // Floats
  float r = 1.2f, g = 3.0f, b = -12.0f;
  clamp(r, g, b);
  assert_eq(r, 1.0f);
  assert_eq(g, 1.0f);
  assert_eq(b, 0.0f);

  r = 0.01f;
  g = 0.8f;
  b = 0.9f;
  clamp(r, g, b);
  assert_eq(r, 0.01f);
  assert_eq(g, 0.8f);
  assert_eq(b, 0.9f);

  return result;
}

// Test the uv functions
bool uv_image(ostream& outstream, ostream& errstream){
  bool result = true;
  
  Image img = Image(16, 24, 2);

  for(int i = 0; i < 16; i++){
    assert_eq(img.u(i), float(i)/15);
  }

  for(int i = 0; i < 24; i++){
    assert_eq(img.v(i), float(i)/23);
  }

  return result;
}

// test px, py and ps
bool idxs_image(ostream& outstream, ostream& errstream){
  bool result = true;

  Image img = Image(8, 6, 4);

  for(int y = 0; y < 6; y++){
    for(int x = 0; x < 8; x++){
      for(int s = 0; s < 4; s++){

        int idx = 4*((8*y+x))+s;
        outstream << "idx = " << idx << "\n";
        errstream << "idx = " << idx << "\n";
        assert_eq(img.px(idx), x);
        assert_eq(img.py(idx), y);
        assert_eq(img.ps(idx), s);

      }
    }
  }

  return result;
}

// Test the raster updater
/*bool raster_update_image(ostream& outstream, ostream& errstream){
  bool result = true;

  // Image Test no thread samples, just updating with some rays.
  Image img_basic(2, 2, 1);
  ThreadMem threads_basic[2*2];
  threads_basic[0].color = {1.0f, 1.0f, 1.0f};
  threads_basic[1].color = {1.0f, 0.0f, 0.0f};
  threads_basic[2].color = {0.0f, 1.0f, 0.0f};
  threads_basic[3].color = {0.0f, 0.0f, 1.0f};
  threads_basic[0].t = 1.0f;
  threads_basic[1].t = 1.0f;
  threads_basic[2].t = 1.0f;
  threads_basic[3].t = 1.0f;

  img_basic.update_raster(&threads_basic[0]);

  // Check that the image updated correctly.
  //float *raster = img_basic.raster;
  string output = "P3\n2 2\n255\n";
  output += "255 255 255\n";
  output += "255 0 0\n";
  output += "0 255 0\n";
  output += "0 0 255\n";
  // Now we would like to average to another color.
  compare_imgs(img_basic, output);

  threads_basic[0].color = {0.0f, 0.0f, 0.0f};
  threads_basic[1].color = {0.0f, 0.0f, 0.0f};
  threads_basic[2].color = {0.0f, 0.0f, 0.0f};
  threads_basic[3].color = {0.0f, 0.0f, 0.0f};
  threads_basic[0].t = -1.0f;
  threads_basic[2].t = -1.0f;
  img_basic.update_raster(&threads_basic[0]);

  output = "P3\n2 2\n255\n";
  output += "127 127 127\n";
  output += "127 0 0\n";
  output += "0 127 0\n";
  output += "0 0 127\n";

  compare_imgs(img_basic, output);

  threads_basic[1].t = -1.0f;
  threads_basic[3].t = -1.0f;

  img_basic.update_raster(&threads_basic[0]);
  output = "P3\n2 2\n255\n";
  output += "127 127 127\n";
  output += "85 0 0\n";
  output += "0 127 0\n";
  output += "0 0 85\n";

  compare_imgs(img_basic, output);
  img_basic.update_raster(&threads_basic[0]);
  img_basic.update_raster(&threads_basic[0]);
  compare_imgs(img_basic, output);

  return result;
}*/

// Test the image writing.
bool write_image(ostream& outstream, ostream& errstream){
  bool result = true;

  Image img(6, 4, 4);
  float* raster = img.get_raster();
  
  string output = "P3\n6 4\n255\n";
  // (0, 0) is in the upper left hand corner.
  for(int i = 0; i < 6*4; i++){
    float r = (float(i+0))/(6*4*3);
    float g = (float(i+1))/(6*4*3);
    float b = (float(i+2))/(6*4*3);

    int cr = (int)(r*255);
    int cg = (int)(g*255);
    int cb = (int)(b*255);

    raster[(i*3)+0] = r;
    raster[(i*3)+1] = g;
    raster[(i*3)+2] = b;

    output += std::to_string(cr) + " " + std::to_string(cg) + " ";
    output += std::to_string(cb) + "\n";
  }

  compare_imgs(img, output);
  img.writePPMtofile("imgs/test_write_image.ppm");

  return result;
}

bool read_image(ostream& outstream, ostream& errstream){
  bool result = true;

  Image img(6, 4, 4);
  float* raster = img.get_raster();
  
  string output = "P3\n6 4\n255\n";
  // (0, 0) is in the upper left hand corner.
  for(int i = 0; i < 6*4; i++){
    float r = (float(i+0))/(6*4*3);
    float g = (float(i+1))/(6*4*3);
    float b = (float(i+2))/(6*4*3);

    int cr = (int)(r*255);
    int cg = (int)(g*255);
    int cb = (int)(b*255);

    raster[(i*3)+0] = r;
    raster[(i*3)+1] = g;
    raster[(i*3)+2] = b;

    output += std::to_string(cr) + " " + std::to_string(cg) + " ";
    output += std::to_string(cb) + "\n";
  }

  img.writePPMtofile("imgs/test_read_image.ppm");

  Image img2(6, 4, 4);
  img2.readPPM("imgs/test_read_image.ppm");
  compare_imgs(img2, output);

  return result;
}
// Test the raster updater
/*bool raster_update_image2(ostream& outstream, ostream& errstream){
  bool result = true;

  // Image Test no thread samples, just updating with some rays.
  Image img(2, 2, 2);
  ThreadMem threads_basic[2*2*2];
  threads_basic[0].color = {1.0f, 1.0f, 1.0f};
  threads_basic[1].color = {0.0f, 0.0f, 0.0f};

  threads_basic[2].color = {1.0f, 0.0f, 0.0f};
  threads_basic[3].color = {0.0f, 1.0f, 0.0f};

  threads_basic[4].color = {0.0f, 1.0f, 0.0f};
  threads_basic[5].color = {0.0f, 0.0f, 1.0f};
  
  threads_basic[6].color = {0.0f, 0.0f, 1.0f};
  threads_basic[7].color = {1.0f, 0.0f, 0.0f};

  for(int i = 0; i < 8; i++){
    threads_basic[i].t = 1.0f;
  }

  img.update_raster(&threads_basic[0]);

  // Check that the image updated correctly.
  string output = "P3\n2 2\n255\n";
  output += "127 127 127\n";
  output += "127 127 0\n";
  output += "0 127 127\n";
  output += "127 0 127\n";
  // Now we would like to average to another color.
  compare_imgs(img, output);

  //Image img2(2, 2, 2);
  for(int i = 0; i < 8; i++){
    if (i%2 == 0)
      threads_basic[i].color = {1.0f, 0.0f, 0.0f};
    else
      threads_basic[i].color = {0.0f, 1.0f, 0.0f};
  }
  threads_basic[0].t = -1.0f;
  threads_basic[1].t = -1.0f;
  threads_basic[2].t = -1.0f;
  threads_basic[5].t = -1.0f;
  img.update_raster(&threads_basic[0]);

  output = "P3\n2 2\n255\n";
  output += "127 127 63\n";
  output += "127 127 0\n";
  output += "63 127 63\n";
  output += "127 63 63\n";

  compare_imgs(img, output);

  img.update_raster(&threads_basic[0]);
  output = "P3\n2 2\n255\n";
  output += "127 127 63\n";
  output += "85 170 0\n";
  output += "127 85 42\n";
  output += "127 85 42\n";

  compare_imgs(img, output);

  return result;
}*/

// UTILITY FUNCTION -----------------------------------------------
// Compares two images.
void compare_imgs_mac(Image& img, string& output, ostream& outstream, 
    ostream& errstream, bool &result){

  // Get the image
  ostringstream outimg;
  img.writePPM(outimg);

  // Check
  result = (outimg.str() == output) && result;

  // Show results
  if(!result){
    errstream << "OUT\n" << outimg.str() << "\nCALC\n" << output << "\n";
  }
  outstream << "OUT\n" << outimg.str() << "\nCALC\n" << output << "\n";
}

