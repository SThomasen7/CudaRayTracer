#include "asset.h"
#include <stdexcept>
#include <limits>
#include <algorithm>

using std::string;
using std::vector;

float dist(float* a, float* b);
// Creates one memory manager per mesh
void create_memory_manager(string filename, TriangleMemoryManager*** managers,
                            size_t& num_managers, int material_index){

    // Bounding sphere is null coming in
    // it should have four floats, vert + radius

    //cout << "Start loading model" << endl;
    Assimp::Importer importer;
    const aiScene *scene = importer.ReadFile(filename, 0);
    aiMesh** meshes = scene->mMeshes;
    num_managers = scene->mNumMeshes;

    // Check if the file name can be opened.
    if(scene == nullptr){
      throw std::runtime_error("Error reading mesh file.");
    }


    *managers = new TriangleMemoryManager*[num_managers];

    for(size_t i = 0; i < num_managers; i++){
        aiMesh* mesh = meshes[i];

        float* verts = new float[mesh->mNumVertices*3];

        float max_x = -std::numeric_limits<float>::infinity();
        float max_y = -std::numeric_limits<float>::infinity();
        float max_z = -std::numeric_limits<float>::infinity();
        float min_x = std::numeric_limits<float>::infinity();
        float min_y = std::numeric_limits<float>::infinity();
        float min_z = std::numeric_limits<float>::infinity();
        // Copy over vertex memory for this mesh
        for(size_t vert_idx = 0; vert_idx < mesh->mNumVertices; vert_idx++){
          aiVector3D vec = mesh->mVertices[vert_idx];
          verts[(vert_idx*3)+0] = vec.x;
          verts[(vert_idx*3)+1] = vec.y;
          verts[(vert_idx*3)+2] = vec.z;

          max_x = std::max(vec.x, max_x);
          max_y = std::max(vec.y, max_y);
          max_z = std::max(vec.z, max_z);
          min_x = std::min(vec.x, min_x);
          min_y = std::min(vec.y, min_y);
          min_z = std::min(vec.z, min_z);
        }

        float bounding_sphere[4];
        bounding_sphere[0] = (max_x + min_x) / 2.0f;
        bounding_sphere[1] = (max_y + min_y) / 2.0f;
        bounding_sphere[2] = (max_z + min_z) / 2.0f;
        float max_dist = -std::numeric_limits<float>::infinity();
        for(size_t vert_idx = 0; vert_idx < mesh->mNumVertices; vert_idx++){
          max_dist = std::max(max_dist,  dist(&bounding_sphere[0], 
                                              &verts[vert_idx*3]));
        }
        bounding_sphere[3] = max_dist;

        // Get the radius of the bounding sphere
        // max dist from center to outside vertex
        
        // Count the number of indices needed for this mesh
        size_t indices_count = 0;
        size_t num_faces = 0;
        for(size_t face_idx = 0; face_idx < mesh->mNumFaces; face_idx++){
          aiFace face = mesh->mFaces[face_idx];

          if(face.mNumIndices > 4 || face.mNumIndices < 3){
            throw std::runtime_error(string("Mesh processor cannot triangulate a")+
                string(" face with more than four sides or less than three sides."));
          }

          if(face.mNumIndices == 3){
            indices_count += 3;
            num_faces += 1;
          }
          else{
            indices_count += 6;
            num_faces += 2;
          }
        }

        // Load the index data
        float* indices = new float[indices_count];
        indices_count = 0;
        for(size_t face_idx = 0; face_idx < mesh->mNumFaces; face_idx++){
          aiFace face = mesh->mFaces[face_idx];

          if(face.mNumIndices == 3){
            indices[indices_count+0] = (float)face.mIndices[0];
            indices[indices_count+1] = (float)face.mIndices[1];
            indices[indices_count+2] = (float)face.mIndices[2];
            indices_count += 3;
          }
          // We need to split squares into two triangles
          else{
            indices[indices_count+0] = (float)face.mIndices[0];
            indices[indices_count+1] = (float)face.mIndices[1];
            indices[indices_count+2] = (float)face.mIndices[2];

            indices[indices_count+3] = (float)face.mIndices[2];
            indices[indices_count+4] = (float)face.mIndices[3];
            indices[indices_count+5] = (float)face.mIndices[0];

            indices_count += 6;
          }

        }

        (*managers)[i] = new TriangleMemoryManager(num_faces,
            mesh->mNumVertices, false, material_index);

        (*managers)[i]->load(&verts[0], &indices[0], &bounding_sphere[0]);
    }

}

float dist(float* a, float* b){
  return sqrt((a[0]-b[0])*(a[0]-b[0])+
              (a[1]-b[1])*(a[1]-b[1])+
              (a[2]-b[2])*(a[2]-b[2]));
}
