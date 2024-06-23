#ifndef _IMAGE_H_
#define _IMAGE_H_ 1
#include<iostream>                                                                                   
#include<fstream>                                                                                    
#include<string>                                                                                     

#include "vec3.h"
#include "thread_mem.h"

class Image{                                                                                         
  
public: 
  Image(){};
  ~Image();

  Image(int width, int height, int num_samples);                                               

  void writePPMtofile(std::string filename);
  void readPPM(std::string filename);
  void writePPM(std::ostream& out);

  inline int width() { return nx; }
  inline int height() { return ny; }                                                          
  inline int samples() { return num_samples; }

  inline float u(int i) { return float(i) / (nx-1); }                                          
  inline float v(int j) { return float(j) / (ny-1); }                                          

  inline float* get_raster(){ return raster; }                                                 

  int px(int id);                                                                         
  int py(int id);                                                                         
  int ps(int id);                                                                         

  void update_raster(ThreadMem* mem);                                                         

  float* raster;
  bool* dead; // track the dead rays                                                                 
  int* num_rays;
  int nx; //raster width
  int ny; //raster height.
  int num_samples;

};


void clamp(float*);
void clamp(Vec3&);
void clamp(float &x, float &y, float &z);

#endif
