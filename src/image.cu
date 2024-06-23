#include "image.h"

#include <cmath>
#include <fstream>                                                                                   
#include <iostream>                                                                                  
#include <cstring>

using namespace std;

void clamp(float* color){
  for(int i = 0; i < 3; i++){
    color[i] = max(color[i], 0.0f);
    color[i] = min(color[i], 1.0f);
  }
} 
  
void clamp(Vec3& color){
  color.x = min(max(color.x, 0.0f), 1.0f);
  color.y = min(max(color.y, 0.0f), 1.0f);
  color.z = min(max(color.z, 0.0f), 1.0f);
} 

void clamp(float &x, float &y, float &z){

  x = min(max(x, 0.0f), 1.0f);
  y = min(max(y, 0.0f), 1.0f);
  z = min(max(z, 0.0f), 1.0f);

}
  
//Initialize raster to default rgb color
Image::Image(int width, int height, int num_samples){
  nx = width;
  ny = height;
  this->num_samples = num_samples;
  
  raster = (float*)malloc(sizeof(float)*width*height*3);
  //num_rays = (int*)malloc(sizeof(int)*width*height);
  //dead = (bool*)malloc(sizeof(bool)*width*height*num_samples);                                       
  
  /*for(int i = 0; i < width*height; i++){
    num_rays[i] = 0;
  }*/
  
  for(int i = 0; i < width*height*3; i++){                                                           
    raster[i] = 0.0f;
  }

  /*for(int i = 0; i < width*height*num_samples; i++){
    dead[i] = false;
  }**/
}


//Free the memory
Image::~Image(){
  free(raster);
  //free(num_rays);
  //free(dead);
}

// Updates the image raster based on thread Mem.
// The dead raster has size nx*ny*num_samples, as well as 
// ray_num.
// Thread mem comes from the gpu, it contains the current ray
// and t value. If t is less than 0, a child ray will not be
// Spawned. I.e. This pixel is complete.
// In this case we add the ray color to the pixel and divide by
// num_rays (when printing the image.)
// We need to average across all of the samples every time we
// add to the pixel color. This is where dead comes in.
void Image::update_raster(ThreadMem* mem){

  for(int i = 0; i < nx; i++){
    for(int j = 0; j < ny; j++){
      float* pixel = &raster[(i+j*nx)*3];
      //int* r_count = &num_rays[i+j*nx];

      Vec3 color = {0.0f, 0.0f, 0.0f};
      //float r, g, b;
      //r = g = b = 0.0f;
      //int div = num_samples;
      for(int s = 0; s < num_samples; s++){
        int idx = ((i+j*nx)*num_samples)+s;
        ThreadMem* cell = &mem[idx];

        //cout << idx << " " << cell->color.x <<" " <<  cell->color.y <<" " <<  cell->color.z << " " << cell->count << endl;
        //color = vec3_add(color, vec3_div(cell->color, cell->count));
        color = vec3_add(color, cell->color);

        //if(dead[idx]){
          //div--;
          //continue;
        //}

        // The ray has not hit anything, but this is
        // the first time, so the background color should
        // still be recorded.
        //if(cell->t < 0.0f){
          //dead[idx] = true;
          //r += cell->color.x;
          //g += cell->color.y;
          //b += cell->color.z;
          //continue;
        //}

        //r += cell->color.x;
        //g += cell->color.y;
        //b += cell->color.z;

      //}
      }

      //if(div > 0){
        //if (div < num_samples && *r_count > 0){
          //r += (num_samples-div)*pixel[0]/(*r_count);
          //g += (num_samples-div)*pixel[1]/(*r_count);
          //b += (num_samples-div)*pixel[2]/(*r_count);
        //}
        //(*r_count)++;
        //r /= div;
        //g /= div;
        //b /= div;

        //cout << "div " << div << endl;
        //pixel[0] += r;
        //pixel[1] += g;
        //pixel[2] += b;
      //}

    vec3_divequal(color, num_samples);
    pixel[0] = color.x;
    pixel[1] = color.y;
    pixel[2] = color.z;
    }
  }
}

void Image::writePPMtofile(string filename){
  ofstream out;
  out.open(filename.c_str());
  if(!out.is_open()){
    std::cerr << " ERROR -- File: " << filename << " could not be opened.\n";
    exit(-1);
  }
  writePPM(out);
  out.close();
}

//Outputs PPM image to file
void Image::writePPM(ostream& out){
  //Write the header

  out << "P3\n";
  out << nx << " " << ny << "\n";
  out << "255\n";

  int i, j;

  for(i = 0; i < ny; i++){
    for(j = 0; j < nx; j++){
      float* pixel = &raster[(j+nx*i)*3];
      //printf("i: %d, j: %d, idx: %d\n", i, j, (j+nx*i)*3);
      //printf("pixel: (%f, %f, %f)\n", pixel[0], pixel[1], pixel[2]);

      unsigned int red = (255.0f*pixel[0]);
      unsigned int grn = (255.0f*pixel[1]);
      unsigned int blu = (255.0f*pixel[2]);
      //if(num_rays[j+nx*i] > 0){
        //red = (255.0f*pixel[0])/num_rays[(j+nx*i)];
        //grn = (255.0f*pixel[1])/num_rays[(j+nx*i)];
        //blu = (255.0f*pixel[2])/num_rays[(j+nx*i)];
        //red = (255.0f*pixel[0]);
        //grn = (255.0f*pixel[1]);
        //blu = (255.0f*pixel[2]);
      //}

      clamp(pixel);
      out << red << " " << grn << " "
        << blu << "\n";
    }
  }
}

void Image::readPPM(string filename){
  //Open the file
  ifstream in;
  in.open(filename.c_str());
  if(!in.is_open()){
    std::cerr << " ERROR -- File: " << filename << " could not be opened.\n";
    exit(-1);
  }

  char ch, type;
  //int red, green, blue;
  int cols, rows;
  int num;

  //Read in header info
  in.get(ch);
  in.get(type);
  in >> cols >> rows >> num;

  nx = cols;
  ny = rows;

  raster = new float[nx*ny*3];

  //Eat newline
  in.get(ch);

  int color;
  //Read in each PPM pixel
  for(int i = 0; i < nx*ny*3; i++){
      in >> color;
      raster[i] = (float)((unsigned char) color)/255.0;
  }
}


int Image::px(int id){
  return (id/num_samples) % nx;
}

int Image::py(int id){
  return (id/num_samples) / nx;
}

int Image::ps(int id){
  return (id % num_samples);
}



