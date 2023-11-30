/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/
#ifndef _PETRINET_KERNEL_H_
#define _PETRINET_KERNEL_H_

#include <stdio.h>

#define MAX_DEVICE_MEM 2000000000
#define BLOCK_SIZE 256
#define BLOCK_SIZE_BITS 8

#include "../header/pets_common.h"
#define PNS_GRID_DIM (SM_NUM * 1)

__device__ void ori_fire_transition(char* g_places, int* conflict_array, int tr, 
		     int tc, int step, int N, int thd_thrd) 
{
	int val1, val2, val3, to_update;
	int mark1, mark2;

	to_update = 0;
	if (threadIdx.x<thd_thrd) 
	{
		// check if the transition is enabled and conflict-free
		val1 = (tr==0)? (N+N)-1: tr-1;
		val2 = (tr & 0x1)? (tc==N-1? 0: tc+1): tc;
		val3 = (tr==(N+N)-1)? 0: tr+1;
		mark1 = g_places[val1*N+val2];
		mark2 = g_places[tr*N+tc];
		if ( (mark1>0) && (mark2>0) ) 
		{
			to_update = 1;
			conflict_array[tr*N+tc] = step;
		}
	}
	__syncthreads();

	if (to_update) 
	{
		// If there are conflicts, transitions on even/odd rows are 
		// kept when the step is even/odd
		to_update = ((step & 0x01) == (tr & 0x01) ) || ((conflict_array[val1*N+val2]!=step) && 
			(conflict_array[val3*N+((val2==0)? N-1: val2-1)]!=step));
	}

	// now update state
	// 6 kernel memory accesses 
	if (to_update) 
	{
		g_places[val1*N+val2] = mark1-1;  // the place above
		g_places[tr*N+tc] = mark2-1; // the place on the left
	}
	__syncthreads();
	if (to_update) 
	{
		g_places[val3*N+val2]++;  // the place below
		g_places[tr*N+(tc==N-1? 0: tc+1)]++; // the place on the right
	}
	__syncthreads();
}


__device__ void ori_initialize_grid(int* g_places, int NSQUARE2, int seed) 
{
	// N is an even number
	int i;
	int loop_num = NSQUARE2 >> (BLOCK_SIZE_BITS+2);

	for (i=0; i<loop_num; i++) 
	{
		g_places[threadIdx.x+(i<<BLOCK_SIZE_BITS)] = 0x01010101;
	}

	if (threadIdx.x < (NSQUARE2>>2)-(loop_num<<BLOCK_SIZE_BITS)) 
	{
		g_places[threadIdx.x+(loop_num<<BLOCK_SIZE_BITS)] = 0x01010101;
	}

	ori_RandomInit(blockIdx.x+seed);
}


__device__ void ori_run_trajectory(int* g_places, int N, int max_steps) 
{
	int step, NSQUARE2, val;

	step = 0;
	NSQUARE2 = (N+N)*N;

	while (step<max_steps) 
	{
		ori_BRandom(); // select the next MERS_N (624) transitions

		// process 256 transitions
		val = mt[threadIdx.x]%NSQUARE2;
		ori_fire_transition((char*)g_places, g_places+(NSQUARE2>>2), 
				val/N, val%N, step+7, N, BLOCK_SIZE);
		
		// process 256 transitions
		val = mt[threadIdx.x+BLOCK_SIZE]%NSQUARE2;
		ori_fire_transition((char*)g_places, g_places+(NSQUARE2>>2), 
				val/N, val%N, step+11, N, BLOCK_SIZE);
						
		// process 112 transitions
		if (  threadIdx.x < MERS_N-(BLOCK_SIZE<<1)  ) 
		{
			val = mt[threadIdx.x+(BLOCK_SIZE<<1)]%NSQUARE2;
		}
		ori_fire_transition((char*)g_places, g_places+(NSQUARE2>>2), 
				val/N, val%N, step+13, N, MERS_N-(BLOCK_SIZE<<1));

		step += MERS_N>>1; 
		// experiments show that for N>2000 and max_step<20000, 
		// the step increase is larger than 320
	}
}


__device__ void ori_compute_reward_stat(int* g_places, float* g_vars, int* g_maxs, int NSQUARE2) 
{
	float sum = 0;
	int i;
	int max = 0;
	int temp, data; 
	int loop_num = NSQUARE2 >> (BLOCK_SIZE_BITS+2);
	for (i=0; i<=loop_num-1; i++) 
	{  // a bug. i<loop_num should be changed to i<=loop_num-1
		data = g_places[threadIdx.x+(i<<BLOCK_SIZE_BITS)];
		
		temp = data & 0x0FF;
		sum += temp*temp;
		max = max<temp? temp: max;
		temp = (data>>8) & 0x0FF;
		sum += temp*temp;
		max = max<temp? temp: max;
		temp = (data>>16) & 0x0FF;
		sum += temp*temp;
		max = max<temp? temp: max;
		temp = (data>>24) & 0x0FF;
		sum += temp*temp;
		max = max<temp? temp: max;
	}

	i = NSQUARE2>>2;
	i &= 0x0FF;
	loop_num *= BLOCK_SIZE; 
	// I do not know why loop_num<<=BLOCK_SIZE_BITS does not work
	if (threadIdx.x <= i-1) 
	{
		data = g_places[threadIdx.x+loop_num];
		
		temp = data & 0x0FF;
		sum += temp*temp;
		max = max<temp? temp: max;
		temp = (data>>8) & 0x0FF;
		sum += temp*temp;
		max = max<temp? temp: max;
		temp = (data>>16) & 0x0FF;
		sum += temp*temp;
		max = max<temp? temp: max;
		temp = (data>>24) & 0x0FF;
		sum += temp*temp;
		max = max<temp? temp: max;
	}

	((float*)mt)[threadIdx.x] = (float)sum;
	mt[threadIdx.x+BLOCK_SIZE] = (uint32)max;
	__syncthreads();
		
	for (i=(BLOCK_SIZE>>1); i>0; i = (i>>1) ) 
	{
		if (threadIdx.x<i) 
		{
			((float*)mt)[threadIdx.x] += ((float*)mt)[threadIdx.x+i];
			if (mt[threadIdx.x+BLOCK_SIZE]<mt[threadIdx.x+i+BLOCK_SIZE])
			mt[threadIdx.x+BLOCK_SIZE] = mt[threadIdx.x+i+BLOCK_SIZE];
		}
		__syncthreads();
	}
		
	if (threadIdx.x==0) 
	{
		g_vars[blockIdx.x] = (((float*)mt)[0])/NSQUARE2-1; 
		// D(X)=E(X^2)-E(X)^2, E(X)=1
		g_maxs[blockIdx.x] = (int)mt[BLOCK_SIZE];
	}
}

// Kernel function for simulating Petri Net for a defined grid
// n: the grid has 2nX2n places and transitions together
// s: steps in each trajectory
// t: number of trajectories
__global__ void ori_pns(int* g_s, float* g_v, int* g_m, int n, int s, int seed, int iteration) 
{
	for (int loop = 0; loop < iteration; loop++) {
		// block size must be 256
		// n is an even number
		int NSQUARE2 = n*n*2;
		int* g_places = g_s+blockIdx.x*((NSQUARE2>>2)+NSQUARE2);   
		// place numbers, conflict_array
		ori_initialize_grid(g_places, NSQUARE2, seed);

		ori_run_trajectory(g_places, n, s);
		ori_compute_reward_stat(g_places, g_v, g_m, NSQUARE2);
	}
}


__device__ void ptb_fire_transition(char* g_places, int* conflict_array, int tr, 
		     int tc, int step, int N, int thd_thrd,
             int thread_id_x, int block_id_x) 
{
	int val1, val2, val3, to_update;
	int mark1, mark2;

	to_update = 0;
	if (thread_id_x<thd_thrd) 
	{
		// check if the transition is enabled and conflict-free
		val1 = (tr==0)? (N+N)-1: tr-1;
		val2 = (tr & 0x1)? (tc==N-1? 0: tc+1): tc;
		val3 = (tr==(N+N)-1)? 0: tr+1;
		mark1 = g_places[val1*N+val2];
		mark2 = g_places[tr*N+tc];
		if ( (mark1>0) && (mark2>0) ) 
        {
            to_update = 1;
            conflict_array[tr*N+tc] = step;
        }
	}
	__syncthreads();

	if (to_update) 
	{
		// If there are conflicts, transitions on even/odd rows are 
		// kept when the step is even/odd
		to_update = ((step & 0x01) == (tr & 0x01) ) || ((conflict_array[val1*N+val2]!=step) && 
			(conflict_array[val3*N+((val2==0)? N-1: val2-1)]!=step));
	}

	// now update state
	// 6 kernel memory accesses 
	if (to_update) 
	{
		g_places[val1*N+val2] = mark1-1;  // the place above
		g_places[tr*N+tc] = mark2-1; // the place on the left
	}
	__syncthreads();
	if (to_update) 
	{
		g_places[val3*N+val2]++;  // the place below
		g_places[tr*N+(tc==N-1? 0: tc+1)]++; // the place on the right
	}
	__syncthreads();
}


__device__ void ptb_initialize_grid(int* g_places, int NSQUARE2, int seed,
                int thread_id_x, int block_id_x) 
{
	// N is an even number
	int i;
	int loop_num = NSQUARE2 >> (BLOCK_SIZE_BITS+2);

	for (i=0; i<loop_num; i++) 
	{
		g_places[thread_id_x+(i<<BLOCK_SIZE_BITS)] = 0x01010101;
	}

	if (thread_id_x < (NSQUARE2>>2)-(loop_num<<BLOCK_SIZE_BITS)) 
	{
		g_places[thread_id_x+(loop_num<<BLOCK_SIZE_BITS)] = 0x01010101;
	}

	ptb_RandomInit(block_id_x+seed, thread_id_x);
}


__device__ void ptb_run_trajectory(int* g_places, int N, int max_steps,
                int thread_id_x, int block_id_x) 
{
	int step, NSQUARE2, val;

	step = 0;
	NSQUARE2 = (N+N)*N;

	while (step<max_steps) 
	{
		ptb_BRandom(thread_id_x); // select the next MERS_N (624) transitions

		// process 256 transitions
		val = mt[thread_id_x]%NSQUARE2;
		ptb_fire_transition((char*)g_places, g_places+(NSQUARE2>>2), 
				val/N, val%N, step+7, N, BLOCK_SIZE, 
                thread_id_x, block_id_x);
		
		// process 256 transitions
		val = mt[thread_id_x+BLOCK_SIZE]%NSQUARE2;
		ptb_fire_transition((char*)g_places, g_places+(NSQUARE2>>2), 
				val/N, val%N, step+11, N, BLOCK_SIZE, 
                thread_id_x, block_id_x);
						
		// process 112 transitions
		if (  thread_id_x < MERS_N-(BLOCK_SIZE<<1)  ) 
		{
			val = mt[thread_id_x+(BLOCK_SIZE<<1)]%NSQUARE2;
		}
		ptb_fire_transition((char*)g_places, g_places+(NSQUARE2>>2), 
				val/N, val%N, step+13, N, MERS_N-(BLOCK_SIZE<<1), 
                thread_id_x, block_id_x);

		step += MERS_N>>1; 
		// experiments show that for N>2000 and max_step<20000, 
		// the step increase is larger than 320
	}
}


__device__ void ptb_compute_reward_stat(int* g_places, float* g_vars, int* g_maxs, int NSQUARE2,
                int thread_id_x, int block_id_x) 
{
	float sum = 0;
	int i;
	int max = 0;
	int temp, data; 
	int loop_num = NSQUARE2 >> (BLOCK_SIZE_BITS+2);
	for (i=0; i<=loop_num-1; i++) 
	{  // a bug. i<loop_num should be changed to i<=loop_num-1
		data = g_places[thread_id_x+(i<<BLOCK_SIZE_BITS)];
		
		temp = data & 0x0FF;
		sum += temp*temp;
		max = max<temp? temp: max;
		temp = (data>>8) & 0x0FF;
		sum += temp*temp;
		max = max<temp? temp: max;
		temp = (data>>16) & 0x0FF;
		sum += temp*temp;
		max = max<temp? temp: max;
		temp = (data>>24) & 0x0FF;
		sum += temp*temp;
		max = max<temp? temp: max;
	}

	i = NSQUARE2>>2;
	i &= 0x0FF;
	loop_num *= BLOCK_SIZE; 
	// I do not know why loop_num<<=BLOCK_SIZE_BITS does not work
	if (thread_id_x <= i-1) 
	{
		data = g_places[thread_id_x+loop_num];
		
		temp = data & 0x0FF;
		sum += temp*temp;
		max = max<temp? temp: max;
		temp = (data>>8) & 0x0FF;
		sum += temp*temp;
		max = max<temp? temp: max;
		temp = (data>>16) & 0x0FF;
		sum += temp*temp;
		max = max<temp? temp: max;
		temp = (data>>24) & 0x0FF;
		sum += temp*temp;
		max = max<temp? temp: max;
	}

	((float*)mt)[thread_id_x] = (float)sum;
	mt[thread_id_x+BLOCK_SIZE] = (uint32)max;
	__syncthreads();
		
	for (i=(BLOCK_SIZE>>1); i>0; i = (i>>1) ) 
	{
		if (thread_id_x<i) 
	{
		((float*)mt)[thread_id_x] += ((float*)mt)[thread_id_x+i];
		if (mt[thread_id_x+BLOCK_SIZE]<mt[thread_id_x+i+BLOCK_SIZE])
		mt[thread_id_x+BLOCK_SIZE] = mt[thread_id_x+i+BLOCK_SIZE];
	}
		__syncthreads();
	}
		
	if (thread_id_x==0) 
	{
		g_vars[block_id_x] = (((float*)mt)[0])/NSQUARE2-1; 
		// D(X)=E(X^2)-E(X)^2, E(X)=1
		g_maxs[block_id_x] = (int)mt[BLOCK_SIZE];
	}
}

// Kernel function for simulating Petri Net for a defined grid
// n: the grid has 2nX2n places and transitions together
// s: steps in each trajectory
// t: number of trajectories
__global__ void ptb_pns(int* g_s, float* g_v, int* g_m, int n, int s, int seed,
				int grid_dimension_x, int block_dimension_x, int iteration) 
{
	unsigned int block_pos = blockIdx.x;
    int thread_id_x = threadIdx.x;

	for (;; block_pos += gridDim.x) {
        if (block_pos >= grid_dimension_x) {
            return;
        }

        int block_id_x = block_pos;

		for (int loop = 0; loop < iteration; loop++) {
			// block size must be 256
			// n is an even number
			int NSQUARE2 = n*n*2;
			int* g_places = g_s+block_id_x*((NSQUARE2>>2)+NSQUARE2);   
			// place numbers, conflict_array
			ptb_initialize_grid(g_places, NSQUARE2, seed, thread_id_x, block_id_x);
			ptb_run_trajectory(g_places, n, s, thread_id_x, block_id_x);
			ptb_compute_reward_stat(g_places, g_v, g_m, NSQUARE2, thread_id_x, block_id_x);
		}
	}
}


__device__ void mix_fire_transition(char* g_places, int* conflict_array, int tr, 
		     int tc, int step, int N, int thd_thrd,
             int thread_id_x, int block_id_x) 
{
	int val1, val2, val3, to_update;
	int mark1, mark2;

	to_update = 0;
	if (thread_id_x<thd_thrd) 
	{
		// check if the transition is enabled and conflict-free
		val1 = (tr==0)? (N+N)-1: tr-1;
		val2 = (tr & 0x1)? (tc==N-1? 0: tc+1): tc;
		val3 = (tr==(N+N)-1)? 0: tr+1;
		mark1 = g_places[val1*N+val2];
		mark2 = g_places[tr*N+tc];
		if ( (mark1>0) && (mark2>0) ) 
        {
            to_update = 1;
            conflict_array[tr*N+tc] = step;
        }
	}
	asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(128) : "memory");

	if (to_update) 
	{
		// If there are conflicts, transitions on even/odd rows are 
		// kept when the step is even/odd
		to_update = ((step & 0x01) == (tr & 0x01) ) || ((conflict_array[val1*N+val2]!=step) && 
			(conflict_array[val3*N+((val2==0)? N-1: val2-1)]!=step));
	}

	// now update state
	// 6 kernel memory accesses 
	if (to_update) 
	{
		g_places[val1*N+val2] = mark1-1;  // the place above
		g_places[tr*N+tc] = mark2-1; // the place on the left
	}
	asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(128) : "memory");
	if (to_update) 
	{
		g_places[val3*N+val2]++;  // the place below
		g_places[tr*N+(tc==N-1? 0: tc+1)]++; // the place on the right
	}
	asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(128) : "memory");
}


__device__ void mix_initialize_grid(int* g_places, int NSQUARE2, int seed,
                int thread_id_x, int block_id_x) 
{
	// N is an even number
	int i;
	int loop_num = NSQUARE2 >> (BLOCK_SIZE_BITS+2);

	for (i=0; i<loop_num; i++) 
	{
		g_places[thread_id_x+(i<<BLOCK_SIZE_BITS)] = 0x01010101;
	}

	if (thread_id_x < (NSQUARE2>>2)-(loop_num<<BLOCK_SIZE_BITS)) 
	{
		g_places[thread_id_x+(loop_num<<BLOCK_SIZE_BITS)] = 0x01010101;
	}

	mix_RandomInit(block_id_x+seed, thread_id_x);
}


__device__ void mix_run_trajectory(int* g_places, int N, int max_steps,
                int thread_id_x, int block_id_x) 
{
	int step, NSQUARE2, val;

	step = 0;
	NSQUARE2 = (N+N)*N;

	while (step<max_steps) 
	{
		mix_BRandom(thread_id_x); // select the next MERS_N (624) transitions

		// process 256 transitions
		val = mt[thread_id_x]%NSQUARE2;
		mix_fire_transition((char*)g_places, g_places+(NSQUARE2>>2), 
				val/N, val%N, step+7, N, BLOCK_SIZE, 
                thread_id_x, block_id_x);
		
		// process 256 transitions
		val = mt[thread_id_x+BLOCK_SIZE]%NSQUARE2;
		mix_fire_transition((char*)g_places, g_places+(NSQUARE2>>2), 
				val/N, val%N, step+11, N, BLOCK_SIZE, 
                thread_id_x, block_id_x);
						
		// process 112 transitions
		if (  thread_id_x < MERS_N-(BLOCK_SIZE<<1)  ) 
		{
			val = mt[thread_id_x+(BLOCK_SIZE<<1)]%NSQUARE2;
		}
		mix_fire_transition((char*)g_places, g_places+(NSQUARE2>>2), 
				val/N, val%N, step+13, N, MERS_N-(BLOCK_SIZE<<1), 
                thread_id_x, block_id_x);

		step += MERS_N>>1; 
		// experiments show that for N>2000 and max_step<20000, 
		// the step increase is larger than 320
	}
}


__device__ void mix_compute_reward_stat(int* g_places, float* g_vars, int* g_maxs, int NSQUARE2,
                int thread_id_x, int block_id_x) 
{
	float sum = 0;
	int i;
	int max = 0;
	int temp, data; 
	int loop_num = NSQUARE2 >> (BLOCK_SIZE_BITS+2);
	for (i=0; i<=loop_num-1; i++) 
	{  // a bug. i<loop_num should be changed to i<=loop_num-1
		data = g_places[thread_id_x+(i<<BLOCK_SIZE_BITS)];
		
		temp = data & 0x0FF;
		sum += temp*temp;
		max = max<temp? temp: max;
		temp = (data>>8) & 0x0FF;
		sum += temp*temp;
		max = max<temp? temp: max;
		temp = (data>>16) & 0x0FF;
		sum += temp*temp;
		max = max<temp? temp: max;
		temp = (data>>24) & 0x0FF;
		sum += temp*temp;
		max = max<temp? temp: max;
	}

	i = NSQUARE2>>2;
	i &= 0x0FF;
	loop_num *= BLOCK_SIZE; 
	// I do not know why loop_num<<=BLOCK_SIZE_BITS does not work
	if (thread_id_x <= i-1) 
	{
		data = g_places[thread_id_x+loop_num];
		
		temp = data & 0x0FF;
		sum += temp*temp;
		max = max<temp? temp: max;
		temp = (data>>8) & 0x0FF;
		sum += temp*temp;
		max = max<temp? temp: max;
		temp = (data>>16) & 0x0FF;
		sum += temp*temp;
		max = max<temp? temp: max;
		temp = (data>>24) & 0x0FF;
		sum += temp*temp;
		max = max<temp? temp: max;
	}

	((float*)mt)[thread_id_x] = (float)sum;
	mt[thread_id_x+BLOCK_SIZE] = (uint32)max;
	asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(128) : "memory");
		
	for (i=(BLOCK_SIZE>>1); i>0; i = (i>>1) ) 
	{
		if (thread_id_x<i) 
	{
		((float*)mt)[thread_id_x] += ((float*)mt)[thread_id_x+i];
		if (mt[thread_id_x+BLOCK_SIZE]<mt[thread_id_x+i+BLOCK_SIZE])
		mt[thread_id_x+BLOCK_SIZE] = mt[thread_id_x+i+BLOCK_SIZE];
	}
		asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(128) : "memory");
	}
		
	if (thread_id_x==0) 
	{
		g_vars[block_id_x] = (((float*)mt)[0])/NSQUARE2-1; 
		// D(X)=E(X^2)-E(X)^2, E(X)=1
		g_maxs[block_id_x] = (int)mt[BLOCK_SIZE];
	}
}


__device__ void mix_PetrinetKernel(int* g_s, float* g_v, int* g_m, int n, int s, int seed,
				int grid_dimension_x, int block_dimension_x, int thread_step, int iteration) 
{
	unsigned int block_pos = blockIdx.x;
    int thread_id_x = threadIdx.x - thread_step;

	for (;; block_pos += PNS_GRID_DIM) {
        if (block_pos >= grid_dimension_x) {
            return;
        }

        int block_id_x = block_pos;

		for (int loop = 0; loop < iteration; loop++) {
			// block size must be 256
			// n is an even number
			int NSQUARE2 = n*n*2;
			int* g_places = g_s+block_id_x*((NSQUARE2>>2)+NSQUARE2);   
			// place numbers, conflict_array
			mix_initialize_grid(g_places, NSQUARE2, seed, thread_id_x, block_id_x);
			mix_run_trajectory(g_places, n, s, thread_id_x, block_id_x);
			mix_compute_reward_stat(g_places, g_v, g_m, NSQUARE2, thread_id_x, block_id_x);
		}
	}
}

#endif // #ifndef _PETRINET_KERNEL_H_
