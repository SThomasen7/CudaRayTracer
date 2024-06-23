#ifndef _TRIANGLE_MEMORY_MANAGER_H_
#define _TRIANGLE_MEMORY_MANAGER_H_ 1

#include "const_memory_manager.h"
#include <set>
#include <map>
#include <iostream>

using std::set;
using std::map;
using std::cout;
using std::endl;

class TriangleMemoryManager : public MemoryManager<float>{

public:
  TriangleMemoryManager(size_t num_faces, size_t num_verts, 
      bool has_normals, float material_index) :
    MemoryManager<float>(num_faces*3) {
      this->num_verts = num_verts;
      this->num_groupsv = 0;
      this->loaded = false;
      this->has_normals = has_normals;
      this->vert_mem = (float*)malloc(sizeof(float)*num_verts*3);
      this->curr_group_first_face = 0;
      this->material_index = material_index;
    }

  TriangleMemoryManager(size_t num_faces, size_t num_verts, 
      bool has_normals, float material_index, size_t max_size) :
    MemoryManager<float>(num_faces*3, max_size) {
      this->num_verts = num_verts;
      this->num_groupsv = 0;
      this->loaded = false;
      this->has_normals = has_normals;
      this->vert_mem = (float*)malloc(sizeof(float)*num_verts*3);
      this->curr_group_first_face = 0;
      this->material_index = material_index;
    }

  ~TriangleMemoryManager(){
    free(this->vert_mem);
  }

  size_t num_groups() { return num_groupsv; }

  void load(float* vert_mem, float* index_mem, 
            float* bounding_sphere);
  void unload();
  // In this case, some base class functions have mildly
  // changed meaning
  // shapes_per_group() number of floats per group
  
  bool increment_group_idx();
  void get_current_group_gpu(float* tri_dmem, int* shape_count);
  float* get_current_group(int* shape_count);
  void free_group(float *group_mem);

  float* alloc_gpu_bsphere();

  size_t get_vert_size();

  float material_index;

private:

  void prep_current_group();
  // Vertex memory
  float *vert_mem;
  float num_verts;
  float bounding_sphere[4];

  size_t num_groupsv;
  size_t curr_group_first_face;
  size_t curr_group_vert_count;
  size_t num_faces_in_group;
  bool loaded;
  bool has_normals;
  // Face memory is the already allocated shape_mem
  /* DECLARED IN BASE CLASS
  float* shape_mem;
  float num_shapes;
  */
};

inline size_t TriangleMemoryManager::get_vert_size(){
  int floats_per_vert = 3;
  if(has_normals){
    floats_per_vert += 3;
  }
  return floats_per_vert;
}

inline float* TriangleMemoryManager::alloc_gpu_bsphere(){
  float* dmem;
  CHECK_CUDA_ERROR(cudaMalloc((void**)&dmem, sizeof(float)*4));
  CHECK_CUDA_ERROR(cudaMemcpy(dmem, &bounding_sphere[0],
                  sizeof(float)*4, cudaMemcpyHostToDevice));
  return dmem;

}

inline void TriangleMemoryManager::load(float* vert_memv, float* index_mem,
      float* bounding_sphere){
  memcpy(&(this->bounding_sphere[0]), bounding_sphere, sizeof(float)*4);
  memcpy(vert_mem, vert_memv, sizeof(float)*num_verts*3);
  memcpy(shape_mem, index_mem, sizeof(float)*num_shapes);
  //cout << num_verts << " " << num_shapes << endl;
  int floats_per_vert = get_vert_size();
  //cout << "floats per vert: " << floats_per_vert << endl;

  // Calculate the number of groups
  int groupc = 0;
  set<int> seen;
  // Iterate over all the indices and determine how many
  // groups we will need.
  int avail = shapes_per_group();
  for(size_t i = 0; i < num_shapes; i++){
    //cout << "avail " << avail << endl;

    //For the current index check if already seen, if so, continue
    if(seen.find((int)shape_mem[i]) != seen.end()){
      continue;
    }
    else{
      // Take into account memory needed for this vertex
      if(seen.size() % 3 == 0){
        avail -= 3;
      }
      seen.insert((int)shape_mem[i]);
      avail -= floats_per_vert;
      //cout << i << " " << groupc << " " << avail<< " "  << endl;
      if(avail <= 0){
        // Ensure the whole triangle is included.
        if(i % 3 != 0){
          i -= i % 3;
        }
        groupc += 1;
        avail = shapes_per_group();
        seen = set<int>();
      }
      if(avail < 0){
        avail = shapes_per_group() - floats_per_vert;
      }
    }

  }
  num_groupsv = groupc;
  loaded = true;
  curr_group_first_face = 0;
  curr_group_vert_count = 0;
  rewind();
  prep_current_group();
}

inline void TriangleMemoryManager::unload(){
  loaded = false;
  num_groupsv = 0;
  curr_group_first_face = 0;
  curr_group_vert_count = 0;
  rewind();
}

// This function will:
// Determine how many faces to put in this group
inline void TriangleMemoryManager::prep_current_group(){
  num_faces_in_group = 0;
  int floats_per_vert = get_vert_size();

  set<int> seen;
  for(size_t i = curr_group_first_face; i < num_shapes; i++){
    int avail = shapes_per_group();

    //For the current index, check if vert fits in mem.
    if(seen.find((int)shape_mem[i]) == seen.end()){
      seen.insert((int)shape_mem[i]);
      avail -= floats_per_vert;
      if(avail <= 0){
        // Ensure the whole triangle is included.
        if((i+1) % 3 != 0){
          break;
        }
        else{
          num_faces_in_group++;
          break;
        }
      }
    } // End seen check
    if((i+1) % 3 == 0 && i != curr_group_first_face){
      num_faces_in_group++;
    }
  }
  curr_group_vert_count = seen.size();
  //cout << "Num faces: " << num_faces_in_group << " " << seen.size()<< endl;
}

inline bool TriangleMemoryManager::increment_group_idx(){
  bool val = MemoryManager::increment_group_idx();
  prep_current_group();
  return val;
};

inline float* TriangleMemoryManager::get_current_group(int* face_count){
  // Allocate auxiliary memory.
  *face_count = num_faces_in_group;
  float* aux = (float*)malloc(sizeof(float)*this->group_size);
  // Initialize, this helps with debugging.
  for(size_t i = 0; i < this->group_size; i++){
    aux[i] = -1.0f;
  }

  // Get number of values in vertex
  size_t vert_size = get_vert_size();

  // Map and index variables
  map<int, int> idxmap;

  size_t j = 0;
  size_t start_vertex = num_faces_in_group*3;
  size_t vert_loaded = 0;
  for(size_t i = curr_group_first_face*3; 
      i < (curr_group_first_face+num_faces_in_group)*3;
      i++){

    // First things first we check if the vertex idx is in the map,
    // if so, the vertex has already been copied, and we just place
    // the transformed index in the array.
    if(idxmap.find(shape_mem[i]) != idxmap.end()){
      aux[j] = (float)idxmap[shape_mem[i]];
    }
    else{
      // The vertex has not yet been added, so we will update the 
      // array to contain this index, and hash the index transform
      idxmap[shape_mem[i]] = vert_loaded;

      // Place this val into aux_mem
      aux[j] = (float)idxmap[shape_mem[i]];

      // Copy the vertex into aux mem.
      for(size_t k = 0; k < vert_size; k++){
        aux[(size_t)(start_vertex+(vert_loaded*vert_size)+k)] = 
          vert_mem[(size_t)(shape_mem[i]*vert_size)+k];
      }
      // Increment the number of verticies loaded
      vert_loaded++;
    }

    j++;
  } // for
  
  return aux;
}

inline void TriangleMemoryManager::free_group(float *group_mem){
  free(group_mem);
  group_mem = nullptr;
}

inline void TriangleMemoryManager::get_current_group_gpu(float* dmem, int *face_count){

  float* aux = get_current_group(face_count);

  // Copy aux mem to the GPU 
  CHECK_CUDA_ERROR(cudaMemcpy(dmem, &aux[0], 
        (num_faces_in_group*3+curr_group_vert_count*get_vert_size())*sizeof(float), 
        cudaMemcpyHostToDevice)
    );
  /*cout << "\nPrinting aux mem " << curr_group_vert_count << endl;
  for(size_t i = 0; i < num_faces_in_group*3+curr_group_vert_count*get_vert_size(); i++){
    cout << aux[i] << " ";
  }
  cout << endl << endl;*/

  // Free aux memory
  free_group(aux);
}

#endif
