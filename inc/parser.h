#ifndef _PARSER_H_
#define _PARSER_H_ 1

#include "config.h"
#include "vec3.h"
#include <string>


SystemConfig parse_cl(int argc, char** argv, bool& fail);
Vec3 comma_sep_vec(std::string vec);

#endif
