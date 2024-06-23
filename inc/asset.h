#ifndef _ASSET_H_
#define _ASSET_H_

#include "triangle_memory_manager.h"
#include <assimp/Importer.hpp>      // C++ importer interface
#include <assimp/scene.h>           // Output data structure
#include <vector>
#include <string>

void create_memory_manager(std::string filename, TriangleMemoryManager*** managers, 
    size_t& num, int material_index);

#endif
