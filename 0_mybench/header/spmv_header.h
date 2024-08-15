//TEXTURE memory
#ifndef __CUDACC__
#define __CUDACC__
#endif // !__CUDACC__
// texture<float,1> tex_x_float;

#ifndef __JDS__
#define __JDS__
//constant memory
__constant__ int jds_ptr_int[5000];
__constant__ int sh_zcnt_int[5000];
#endif __JDS__

inline void input_vec(char *fName,float *h_vec,int dim) {
  FILE* fid = fopen(fName, "rb");
  fread (h_vec, sizeof (float), dim, fid);
  fclose(fid);
  
}

#include "pets_common.h"
#define SPMV_GRID_DIM (SM_NUM * 1)