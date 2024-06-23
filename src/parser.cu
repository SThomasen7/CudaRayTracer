#include "parser.h"
#include "config.h"
#include <getopt.h>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <sstream>

using std::string;
using std::stof;

using std::cout;
using std::endl;

// Util funcs
void print_help();

Vec3 comma_sep_vec(string vec){
  
  float vals[3] = {0.0f, 0.0f, 0.0f};
  int ele = 0;
  int start = 0;
  // Parse the string and separate the three numbers into vector
  for(int i = 0; i < vec.length(); i++){
    if (vec[i] == ','){
      vals[ele++] = stof(vec.substr(start, i-start));
      start = i+1;
    }
  }

  vals[ele] = stof(vec.substr(start, vec.length()-start));

  return vec3_factory(vals[0], vals[1], vals[2]);
}


SystemConfig parse_cl(int argc, char** argv, bool& fail){

  fail = false;
  SystemConfig config;

  // Config defaults
  config.camera.position = vec3_factory(0.0f, 0.0f, 0.0f);
  config.camera.lookat = vec3_factory(0.0f, 0.0f, -1.0f);
  config.camera.up = vec3_factory(0.0f, 1.0f, 0.0f);
  config.camera.vfov = 60.0f;

  auto t = std::time(nullptr);
  auto tm = *std::localtime(&t);
  std::ostringstream oss; 
  oss << std::put_time(&tm, "%Y%m%d_%H%M%S") << ".ppm";
  config.filename = oss.str();

  config.ray_samples = 16;
  config.recursion_depth = 50;

  config.width = 640;
  config.height = 640;

  // Options
  const char* const short_opts = "p:l:u:v:s:d:x:y:f:h";
  static struct option long_opts[] =
    {
      { "camera_pos", required_argument, NULL, 'p'},
      { "camera_look", required_argument, NULL, 'l'},
      { "camera_up", required_argument, NULL, 'u'},
      { "vfov", required_argument, NULL, 'v'},
      { "ray_samples", required_argument, NULL, 's'},
      { "recursion_depth", required_argument, NULL, 'd'},
      { "light_depth", required_argument, NULL, 'i'},
      { "img_width", required_argument, NULL, 'x'},
      { "img_height", required_argument, NULL, 'y'},
      { "file", required_argument, NULL, 'f'},
      { "help", no_argument, NULL, 'h'},
      { NULL, no_argument, NULL, 0}
    };

  
  bool light_depth_assigned = false;
  while(1){
    const auto opt = getopt_long(argc, argv, short_opts, long_opts, NULL);

    if(opt == -1){
      break;
    }

    switch(opt){

      case 'p':
        config.camera.position = comma_sep_vec(optarg);
        break;

      case 'l':
        config.camera.lookat = comma_sep_vec(optarg);
        break;

      case 'u':
        config.camera.up = comma_sep_vec(optarg);
        break;

      case 'v':
        config.camera.vfov = std::stof(optarg);
        break;

      case 's':
        config.ray_samples = std::stoi(optarg);
        break;

      case 'd':
        config.recursion_depth = std::stoi(optarg);
        break;

      case 'i':
        config.light_depth = std::stoi(optarg);
        light_depth_assigned = true;
        break;
        
      case 'x':
        config.width = std::stoi(optarg);
        break;

      case 'y':
        config.height = std::stoi(optarg);
        break;

      case 'f':
        config.filename = string(optarg);
        break;

      case 'h':
      case '?':
      default: 
        print_help();
        fail = true;
        return config;
        break;
    }
  }

  if(!light_depth_assigned){
    config.light_depth = config.recursion_depth;
  }

  display_config(config);
  return config;
}


void print_help(){

  cout << 
    "--camera_pos      -p <vec3>     sets camera position\n"
    "--camera_look     -l <vec3>     sets camera look at\n"
    "--camera_up       -u <vec3>     sets camera position\n"
    "--vfov            -v <num>      vertical field of view (degrees)\n"
    "--ray_samples     -s <num>      number of ray samples\n"
    "--recursion_depth -d <num>      number of times rays recurse\n"
    "--light_depth     -i <num>      depth for light pass, default recursion_depth\n"
    "--img_width       -x <num>      img width\n"
    "--img_height      -y <num>      img height\n"
    "--file            -f <string>   image file name (use .ppm)\n"
    "--help            -h            print this helpful message\n"
      << endl;
}
