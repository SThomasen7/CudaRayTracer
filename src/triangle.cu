#include "triangle.h"

__D__ bool tri_hit_func(Ray& ray, HitRecord& hit, Vec3& p1, Vec3& p2, Vec3& p3);
__D__ bool shadow_tri_hit_func(Ray& ray, HitRecord& hit, Vec3& p1, Vec3& p2, 
                               Vec3& p3, float tlim);


__D__ Triangle get_triangle(int index, float* vert_mem, float* index_buffer){
  float* idxs = &index_buffer[index*3];
  Triangle tri;

  tri.v1 = {vert_mem[(int)(idxs[0]*3)+0],
            vert_mem[(int)(idxs[0]*3)+1],
            vert_mem[(int)(idxs[0]*3)+2]};

  tri.v2 = {vert_mem[(int)(idxs[1]*3)+0],
            vert_mem[(int)(idxs[1]*3)+1],
            vert_mem[(int)(idxs[1]*3)+2]};

  tri.v3 = {vert_mem[(int)(idxs[2]*3)+0],
            vert_mem[(int)(idxs[2]*3)+1],
            vert_mem[(int)(idxs[2]*3)+2]};

  return tri;
}

__D__ TriangleN get_triangleN(int index, float* vert_mem, float* index_buffer){
  float* idxs = &index_buffer[index*3];
  TriangleN tri;

  // Vertex 1
  tri.v1 = {vert_mem[(int)(idxs[0]*6)+0],
            vert_mem[(int)(idxs[0]*6)+1],
            vert_mem[(int)(idxs[0]*6)+2]};

  tri.n1 = {vert_mem[(int)(idxs[0]*6)+3],
            vert_mem[(int)(idxs[0]*6)+4],
            vert_mem[(int)(idxs[0]*6)+5]};

  // Vertex 2
  tri.v2 = {vert_mem[(int)(idxs[1]*6)+0],
            vert_mem[(int)(idxs[1]*6)+1],
            vert_mem[(int)(idxs[1]*6)+2]};

  tri.n2 = {vert_mem[(int)(idxs[1]*6)+3],
            vert_mem[(int)(idxs[1]*6)+4],
            vert_mem[(int)(idxs[1]*6)+5]};

  // Vertex 3
  tri.v3 = {vert_mem[(int)(idxs[2]*6)+0],
            vert_mem[(int)(idxs[2]*6)+1],
            vert_mem[(int)(idxs[2]*6)+2]};

  tri.n3 = {vert_mem[(int)(idxs[2]*6)+3],
            vert_mem[(int)(idxs[2]*6)+4],
            vert_mem[(int)(idxs[2]*6)+5]};


  return tri;
}

__D__ bool triangle_hit(Ray& ray, HitRecord& hit, Triangle& triangle){
  if(tri_hit_func(ray, hit, triangle.v1, triangle.v2, triangle.v3)){

    hit.normal = vec3_get_norm(
                    vec3_cross(
                      vec3_sub(triangle.v2, triangle.v1), 
                      vec3_sub(triangle.v3, triangle.v1)
                    )
                  );

    if(vec3_dot(hit.normal, ray.direction) > 0.0f){
      hit.normal = vec3_mlt(hit.normal, -1.0f);
    }
    return true;
  }
  return false;
}

// Shadow hit the triangle
__D__ bool triangle_shadow_hit(Ray& ray, HitRecord& hit, Triangle& triangle, float tlim){
  return shadow_tri_hit_func(ray, hit, triangle.v1, triangle.v2, triangle.v3, tlim);
}

__D__ bool triangle_hitN(Ray& ray, HitRecord& hit, TriangleN& triangle){
  // TODO Check hit and interpolate the normal

}

__D__ bool tri_hit_func(Ray& ray, HitRecord& hit, Vec3& p0, Vec3& p1, Vec3& p2){

    float tval;
    float A = p0.x - p1.x;
    float B = p0.y - p1.y;
    float C = p0.z - p1.z;

    float D = p0.x - p2.x;
    float E = p0.y - p2.y;
    float F = p0.z - p2.z;

    float G = ray.direction.x;
    float H = ray.direction.y;
    float I = ray.direction.z;

    float J = p0.x - ray.origin.x;
    float K = p0.y - ray.origin.y;
    float L = p0.z - ray.origin.z;


    float EIHF = E*I-H*F;
    float GFDI = G*F-D*I;
    float DHEG = D*H-E*G;

    float denom = (A*EIHF + B*GFDI + C*DHEG);
    float beta  = (J*EIHF + K*GFDI + L*DHEG) / denom;

    //cout << "beta " << beta << endl;
    if(beta <= 0.0 || beta >= 1.0f) return false;

    //cout << "A" << endl;
    float AKJB = A*K - J*B;
    float JCAL = J*C - A*L;
    float BLKC = B*L - K*C;

    float gamma = (I*AKJB + H*JCAL + G*BLKC) / denom;
    if(gamma <= 0.0 || beta + gamma >= 1.0) return false;

    //cout << "B" << endl;
    tval = -(F*AKJB + E*JCAL + D*BLKC) / denom;

    if(tval < 0.001 || hit.t < tval){
        return false;
    }

    hit.t = tval;
    /*record.norm = unitVector(cross((p1 - p0), (p2 - p0)));
    if(dot(record.norm, r.direction()) > 0.0f){
      record.norm = -record.norm;
    }*/

    hit.point = vec3_add(ray.origin, 
                    vec3_mlt(ray.direction,
                             tval)
                   );

    return true;

}

__D__ bool shadow_tri_hit_func(Ray& ray, HitRecord& hit, Vec3& p0, Vec3& p1, 
                                Vec3& p2, float tlim){

    float tval;
    float A = p0.x - p1.x;
    float B = p0.y - p1.y;
    float C = p0.z - p1.z;

    float D = p0.x - p2.x;
    float E = p0.y - p2.y;
    float F = p0.z - p2.z;

    float G = ray.direction.x;
    float H = ray.direction.y;
    float I = ray.direction.z;

    float J = p0.x - ray.origin.x;
    float K = p0.y - ray.origin.y;
    float L = p0.z - ray.origin.z;


    float EIHF = E*I-H*F;
    float GFDI = G*F-D*I;
    float DHEG = D*H-E*G;

    float denom = (A*EIHF + B*GFDI + C*DHEG);
    float beta  = (J*EIHF + K*GFDI + L*DHEG) / denom;

    //cout << "beta " << beta << endl;
    if(beta <= 0.0 || beta >= 1.0f) return false;

    //cout << "A" << endl;
    float AKJB = A*K - J*B;
    float JCAL = J*C - A*L;
    float BLKC = B*L - K*C;

    float gamma = (I*AKJB + H*JCAL + G*BLKC) / denom;
    if(gamma <= 0.0 || beta + gamma >= 1.0) return false;

    //cout << "B" << endl;
    tval = -(F*AKJB + E*JCAL + D*BLKC) / denom;

    if(tval < 0.001 || tlim < tval){
        return false;
    }

    return true;
}
