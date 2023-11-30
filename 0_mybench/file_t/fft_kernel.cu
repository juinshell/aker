
inline __device__ void GPU_FFT2(float2* v){
	float2 vt = v[0];
	v[0] = vt + v[1];
	v[1] = vt - v[1];
}

inline __device__ void GPU_FFT2(float2 &v1, float2 &v2) { 
    float2 v0 = v1;  
    v1 = v0 + v2; 
    v2 = v0 - v2; 
}

inline __device__ void GPU_FFT4(float2 &v0,float2 &v1,float2 &v2,float2 &v3) { 
    GPU_FFT2(v0, v2);
    GPU_FFT2(v1, v3);
    v3 = v3 * exp_1_4;
    GPU_FFT2(v0, v1);
    GPU_FFT2(v2, v3);    
}

inline __device__ void GPU_FFT4(float2* v) {
    GPU_FFT4(v[0],v[1],v[2],v[3] );
}

__device__ int GPU_expand(int idxL, int N1, int N2 ){ 
	return (idxL/N1)*N1*N2 + (idxL%N1); 
}      

__device__ void GPU_exchange( float2* v, int stride, int idxD, int incD, 
	int idxS, int incS ){ 
	__shared__ float work[T*R*2];//T*R*2
	float* sr = work;
	float* si = work+T*R;  
	__syncthreads(); 
	for( int r=0; r<R; r++ ) { 
		int i = (idxD + r*incD)*stride; 
		sr[i] = v[r].x;
		si[i] = v[r].y;  
	}   
	__syncthreads(); 

	for( int r=0; r<R; r++ ) { 
		int i = (idxS + r*incS)*stride;     
		v[r] = make_float2(sr[i], si[i]);  
	}        
}      

__device__ void GPU_DoFft(float2* v, int j, int stride=1) { 
	for( int Ns=1; Ns<N; Ns*=R ){ 
		float angle = -2*M_PI*(j%Ns)/(Ns*R); 
		for( int r=0; r<R; r++ ){
			v[r] = v[r]*make_float2(cos(r*angle), sin(r*angle));
		}

		GPU_FFT2( v );

		int idxD = GPU_expand(j,Ns,R); 
		int idxS = GPU_expand(j,N/R,R); 
		GPU_exchange( v,stride, idxD,Ns, idxS,N/R );
	}      
}

__global__ void ori_fft(float2* data, int iteration) {
	float2 *ori_data = data + blockIdx.x*N;
	for (int loop = 0; loop < iteration; loop++) {
		float2 v[R];
		data = ori_data;

		int idxG = threadIdx.x; 
		for (int r=0; r<R; r++) {  
			v[r] = data[idxG + r*T];
		} 
		GPU_DoFft( v, threadIdx.x );  
		for (int r=0; r<R; r++) {
			data[idxG + r*T] = v[r];
		} 
	}
}

__global__ void general_ptb_fft(float2* data, 
	int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z, int step_size){
			unsigned int block_pos = blockIdx.x;
    int thread_id_x = threadIdx.x;

	for (;; block_pos += step_size) {
        if (block_pos >= grid_dimension_x * grid_dimension_y * grid_dimension_z) {
            return;
        }
		int block_id_x = block_pos;

		float2 *ori_data = data + block_id_x * N;
		float2 v[R];
		// data = ori_data;

		int idxG = thread_id_x; 
		for (int r=0; r<R; r++) {  
			v[r] = ori_data[idxG + r*T];
		} 
		GPU_DoFft( v, thread_id_x );  
		for (int r=0; r<R; r++) {
			ori_data[idxG + r*T] = v[r];
		}
	}
}

// step_size == launch param == ptb worker num == SM_NUM * ptb_per_sm_number


__global__ void ptb_fft(float2* data, 
	int grid_dimension_x, int block_dimension_x, int iteration){

	unsigned int block_pos = blockIdx.x;
    int thread_id_x = threadIdx.x;

	for (;; block_pos += gridDim.x) {
        if (block_pos >= grid_dimension_x) {
            return;
        }
		int block_id_x = block_pos;

		float2 *ori_data = data + block_id_x * N;
		for (int loop = 0; loop < iteration; loop++) {
			float2 v[R];
			// data = ori_data;

			int idxG = thread_id_x; 
			for (int r=0; r<R; r++) {  
				v[r] = ori_data[idxG + r*T];
			} 
			GPU_DoFft( v, thread_id_x );  
			for (int r=0; r<R; r++) {
				ori_data[idxG + r*T] = v[r];
			}
		}
	}
}

__global__ void ptb_fft(float2* data, 
	int grid_dimension_x, int block_dimension_x, int iteration){

	unsigned int block_pos = blockIdx.x;
    int thread_id_x = threadIdx.x;

	for (;; block_pos += gridDim.x) {
        if (block_pos >= grid_dimension_x) {
            return;
        }
		int block_id_x = block_pos;

		float2 *ori_data = data + block_id_x * N;
		for (int loop = 0; loop < iteration; loop++) {
			float2 v[R];
			// data = ori_data;

			int idxG = thread_id_x; 
			for (int r=0; r<R; r++) {  
				v[r] = ori_data[idxG + r*T];
			} 
			GPU_DoFft( v, thread_id_x );  
			for (int r=0; r<R; r++) {
				ori_data[idxG + r*T] = v[r];
			}
		}
	}
}

__device__ void mix_GPU_exchange( float2* v, int stride, int idxD, int incD, 
	int idxS, int incS ){ 
	__shared__ float work[T*R*2];//T*R*2
	float* sr = work;
	float* si = work+T*R;  
	// __syncthreads(); 
	asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(128) : "memory");
	for( int r=0; r<R; r++ ) { 
		int i = (idxD + r*incD)*stride; 
		sr[i] = v[r].x;
		si[i] = v[r].y;  
	}   
	asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(128) : "memory");
	// __syncthreads(); 

	for( int r=0; r<R; r++ ) { 
		int i = (idxS + r*incS)*stride;     
		v[r] = make_float2(sr[i], si[i]);  
	}        
}  

__device__ void mix_GPU_DoFft(float2* v, int j, int stride=1) { 
	for( int Ns=1; Ns<N; Ns*=R ){ 
		float angle = -2*M_PI*(j%Ns)/(Ns*R); 
		for( int r=0; r<R; r++ ){
			v[r] = v[r]*make_float2(cos(r*angle), sin(r*angle));
		}

		GPU_FFT2( v );

		int idxD = GPU_expand(j,Ns,R); 
		int idxS = GPU_expand(j,N/R,R); 
		mix_GPU_exchange( v,stride, idxD,Ns, idxS,N/R );
	}      
}


__device__ void mix_fft(float2* data, 
	int grid_dimension_x, int block_dimension_x, int thread_step, int iteration){

	unsigned int block_pos = blockIdx.x;
    int thread_id_x = threadIdx.x - thread_step;

	for (;; block_pos += FFT_GRID_DIM) {
        if (block_pos >= grid_dimension_x) {
            return;
        }
		int block_id_x = block_pos;

		float2 *ori_data = data + block_id_x * N;
		for (int loop = 0; loop < iteration; loop++) {
			float2 v[R];
			// data = ori_data;

			int idxG = thread_id_x; 
			for (int r=0; r<R; r++) {  
				v[r] = ori_data[idxG + r*T];
			} 
			mix_GPU_DoFft( v, thread_id_x );  
			for (int r=0; r<R; r++) {
				ori_data[idxG + r*T] = v[r];
			}
		}
	}
}
