#include "config.h"
#include <iostream>

using std::cout;
using std::endl;

void display_config(const SystemConfig &config){

  CameraConfig camcon = config.camera;
  cout << "*** System Config ***" << endl;
  cout << "Camera Details: " << endl;
  cout << "Position: (" << camcon.position.x << ", ";
  cout << camcon.position.y;
  cout << ", " << camcon.position.z << ")" << endl;
  cout << "Lookat:   (" << camcon.lookat.x << ", ";
  cout << camcon.lookat.y;
  cout << ", " << camcon.lookat.z << ")" << endl;
  cout << "Up:       (" << camcon.up.x << ", ";
  cout << camcon.up.y;
  cout << ", " << camcon.up.z << ")" << endl;
  cout << "Vfov: " << camcon.vfov << endl;
  cout << "Out file: " << config.filename << endl;
  cout << "Ray Samples: " << config.ray_samples << endl;
  cout << "Depth:       " << config.recursion_depth << endl;
  cout << "Light Depth: " << config.light_depth << endl;
  cout << "Width:       " << config.width << endl;
  cout << "Height:      " << config.height << endl;
  cout << "*********************" << endl;

}
