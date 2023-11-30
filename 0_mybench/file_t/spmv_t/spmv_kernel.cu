


__global__ void ori_spmv(float *dst_vector,
						const float *d_data,const int *d_index, const int *d_perm,
						const float *x_vec,const int *d_nzcnt, const int dim, int iteration) {
    for (int loop = 0; loop < iteration; loop++) {
        int ix=blockIdx.x*blockDim.x+threadIdx.x;
        int warp_id=ix>>5;
        if(ix<dim)
        {
            float sum=0.0f;
            int	bound=sh_zcnt_int[warp_id];
            //prefetch 0
            int j=jds_ptr_int[0]+ix;  
            float d = d_data[j]; 
            int i = d_index[j];  
            float t = x_vec[i];
            
            if (bound>1)  //bound >=2
            {
                //prefetch 1
                j=jds_ptr_int[1]+ix;    
                i =  d_index[j];  
                int in;
                float dn;
                float tn;
                for(int k=2;k<bound;k++ )
                {	
                    //prefetch k-1
                    dn = d_data[j]; 
                    //prefetch k
                    j=jds_ptr_int[k]+ix;    
                    in = d_index[j]; 
                    //prefetch k-1
                    tn = x_vec[i];
                    
                    //compute k-2
                    sum += d*t; 
                    //sweep to k
                    i = in;  
                    //sweep to k-1
                    d = dn;
                    t =tn; 
                }	
            
                //fetch last
                dn = d_data[j];
                tn = x_vec[i];
        
                //compute last-1
                sum += d*t; 
                //sweep to last
                d=dn;
                t=tn;
            }
            //compute last
            sum += d*t;  // 3 3
            
            //write out data
            dst_vector[d_perm[ix]]=sum; 
        }
    }
}


__global__ void ptb_spmv(float *dst_vector,
						const float *d_data,const int *d_index, const int *d_perm,
						const float *x_vec,const int *d_nzcnt, const int dim,
                        int grid_dimension_x, int grid_dimension_y, int block_dimension_x, int block_dimension_y, int iteration) {
    unsigned int block_pos = blockIdx.x;
    int thread_id_x = threadIdx.x % block_dimension_x;
    // int thread_id_y = threadIdx.x / block_dimension_x;

    for (;; block_pos += gridDim.x) {
        if (block_pos >= grid_dimension_x * grid_dimension_y) {
            return;
        }

        for (int loop = 0; loop < iteration; loop++) {

            int block_id_x = block_pos % grid_dimension_x;
            // int block_id_y = block_pos / grid_dimension_x;

            int ix=block_id_x * block_dimension_x + thread_id_x;
            int warp_id = ix>>5;
            if(ix<dim)
            {
                float sum=0.0f;
                int	bound=sh_zcnt_int[warp_id];
                //prefetch 0
                int j=jds_ptr_int[0]+ix;  
                float d = d_data[j]; 
                int i = d_index[j];  
                float t = x_vec[i];
                
                if (bound>1)  //bound >=2
                {
                    //prefetch 1
                    j=jds_ptr_int[1]+ix;    
                    i =  d_index[j];  
                    int in;
                    float dn;
                    float tn;
                    for(int k=2;k<bound;k++ )
                    {	
                        //prefetch k-1
                        dn = d_data[j]; 
                        //prefetch k
                        j=jds_ptr_int[k]+ix;    
                        in = d_index[j]; 
                        //prefetch k-1
                        tn = x_vec[i];
                        
                        //compute k-2
                        sum += d*t; 
                        //sweep to k
                        i = in;  
                        //sweep to k-1
                        d = dn;
                        t =tn; 
                    }	
                
                    //fetch last
                    dn = d_data[j];
                    tn = x_vec[i];
            
                    //compute last-1
                    sum += d*t; 
                    //sweep to last
                    d=dn;
                    t=tn;
                }
                //compute last
                sum += d*t;  // 3 3
                
                //write out data
                dst_vector[d_perm[ix]]=sum; 
            }
        }
    }
}


__device__ void mix_spmv(float *dst_vector,
						const float *d_data,const int *d_index, const int *d_perm,
						const float *x_vec,const int *d_nzcnt, const int dim,
                        int grid_dimension_x, int grid_dimension_y, int block_dimension_x, int block_dimension_y,
                        int thread_step, int iteration) {
    unsigned int block_pos = blockIdx.x;
    int thread_id_x = (threadIdx.x - thread_step) % block_dimension_x;
    // int thread_id_y = (threadIdx.x - thread_step) / block_dimension_x;

    for (;; block_pos += SPMV_GRID_DIM) {
        if (block_pos >= grid_dimension_x * grid_dimension_y) {
            return;
        }

        int block_id_x = block_pos % grid_dimension_x;
        // int block_id_y = block_pos / grid_dimension_x;

        for (int loop = 0; loop < iteration; loop++) {
            int ix=block_id_x * block_dimension_x + thread_id_x;
            int warp_id = ix>>5;
            if(ix<dim)
            {
                float sum=0.0f;
                int	bound=sh_zcnt_int[warp_id];
                //prefetch 0
                int j=jds_ptr_int[0]+ix;  
                float d = d_data[j]; 
                int i = d_index[j];  
                float t = x_vec[i];
                
                if (bound>1)  //bound >=2
                {
                    //prefetch 1
                    j=jds_ptr_int[1]+ix;    
                    i =  d_index[j];  
                    int in;
                    float dn;
                    float tn;
                    for(int k=2;k<bound;k++ )
                    {	
                        //prefetch k-1
                        dn = d_data[j]; 
                        //prefetch k
                        j=jds_ptr_int[k]+ix;    
                        in = d_index[j]; 
                        //prefetch k-1
                        tn = x_vec[i];
                        
                        //compute k-2
                        sum += d*t; 
                        //sweep to k
                        i = in;  
                        //sweep to k-1
                        d = dn;
                        t =tn; 
                    }	
                
                    //fetch last
                    dn = d_data[j];
                    tn = x_vec[i];
            
                    //compute last-1
                    sum += d*t; 
                    //sweep to last
                    d=dn;
                    t=tn;
                }
                //compute last
                sum += d*t;  // 3 3
                
                //write out data
                dst_vector[d_perm[ix]]=sum; 
            }
        }
    }
}
