
__global__ void ori_regtil(float c0,float c1,float *A0,float *Anext, int nx, int ny, int nz, int iteration) {
    for (int loop = 0; loop < iteration; loop++) {

        int i = blockIdx.x*blockDim.x+threadIdx.x;
        int j = blockIdx.y*blockDim.y+threadIdx.y;
        
        float bottom=A0[Index3D (nx, ny, i, j, 0)] ;
        float current=A0[Index3D (nx, ny, i, j, 1)] ;
        if( i>0 && j>0 &&(i<nx-1) &&(j<ny-1) )
        {
            for(int k=1;k<nz-1;k++)
            {
                float top =A0[Index3D (nx, ny, i, j, k+1)] ;
                
                Anext[Index3D (nx, ny, i, j, k)] = 
                (top +
                bottom +
                A0[Index3D (nx, ny, i, j + 1, k)] +
                A0[Index3D (nx, ny, i, j - 1, k)] +
                A0[Index3D (nx, ny, i + 1, j, k)] +
                A0[Index3D (nx, ny, i - 1, j, k)])*c1
                - current*c0;
                bottom=current;
                current=top;
            }
        }
    }
}


__global__ void ptb_regtil(float c0,float c1,float *A0,float *Anext, int nx, int ny, int nz,
    int grid_dimension_x, int grid_dimension_y, int block_dimension_x, int block_dimension_y, int iteration) {
    unsigned int block_pos = blockIdx.x;
    int thread_id_x = threadIdx.x % block_dimension_x;
    int thread_id_y = threadIdx.x / block_dimension_x;

    for (;; block_pos += gridDim.x) {
        if (block_pos >= grid_dimension_x * grid_dimension_y) {
            return;
        }

        int block_id_x = block_pos % grid_dimension_x;
        int block_id_y = block_pos / grid_dimension_x;

        for (int loop = 0; loop < iteration; loop++) {

            int i = block_id_x * block_dimension_x + thread_id_x;
            int j = block_id_y * block_dimension_y + thread_id_y;
            
            float bottom=A0[Index3D (nx, ny, i, j, 0)] ;
            float current=A0[Index3D (nx, ny, i, j, 1)] ;
            if( i>0 && j>0 &&(i<nx-1) &&(j<ny-1) )
            {
                for(int k=1;k<nz-1;k++)
                {
                    float top =A0[Index3D (nx, ny, i, j, k+1)] ;
                    
                    Anext[Index3D (nx, ny, i, j, k)] = 
                    (top +
                    bottom +
                    A0[Index3D (nx, ny, i, j + 1, k)] +
                    A0[Index3D (nx, ny, i, j - 1, k)] +
                    A0[Index3D (nx, ny, i + 1, j, k)] +
                    A0[Index3D (nx, ny, i - 1, j, k)])*c1
                    - current*c0;
                    bottom=current;
                    current=top;
                }
            }
        }
    }
}


__device__ void mix_regtil(float c0,float c1, float *A0, float *Anext, int nx, int ny, int nz,
    int grid_dimension_x, int grid_dimension_y, int block_dimension_x, int block_dimension_y,
    int thread_step, int iteration) {
    unsigned int block_pos = blockIdx.x;
    int thread_id_x = (threadIdx.x - thread_step) % block_dimension_x;
    int thread_id_y = (threadIdx.x - thread_step) / block_dimension_x;

    for (;; block_pos += REGTIL_GRID_DIM) {
        if (block_pos >= grid_dimension_x * grid_dimension_y) {
            return;
        }

        int block_id_x = block_pos % grid_dimension_x;
        int block_id_y = block_pos / grid_dimension_x;

        for (int loop = 0; loop < iteration; loop++) {
            int i = block_id_x * block_dimension_x + thread_id_x;
            int j = block_id_y * block_dimension_y + thread_id_y;
            
            float bottom=A0[Index3D (nx, ny, i, j, 0)] ;
            float current=A0[Index3D (nx, ny, i, j, 1)] ;
            if( i>0 && j>0 &&(i<nx-1) &&(j<ny-1) )
            {
                for(int k=1;k<nz-1;k++)
                {
                    float top =A0[Index3D (nx, ny, i, j, k+1)] ;
                    
                    Anext[Index3D (nx, ny, i, j, k)] = 
                    (top +
                    bottom +
                    A0[Index3D (nx, ny, i, j + 1, k)] +
                    A0[Index3D (nx, ny, i, j - 1, k)] +
                    A0[Index3D (nx, ny, i + 1, j, k)] +
                    A0[Index3D (nx, ny, i - 1, j, k)])*c1
                    - current*c0;
                    bottom=current;
                    current=top;
                }
            }
        }
    }
}

