// #include "pns_randomc.h"

#ifndef RANDOMC_H
#define RANDOMC_H

#include <math.h>          // default math function linrary

#include <assert.h>
#include <stdio.h>

// Define 32 bit signed and unsigned integers.
// Change these definitions, if necessary, on 64 bit computers
typedef   signed int int32; 
typedef unsigned int uint32; 

// constants for MT19937:
#define MERS_N   624
#define MERS_M   397
#define MERS_R   31
#define MERS_U   11
#define MERS_S   7
#define MERS_T   15
#define MERS_L   18
#define MERS_A   0x9908B0DF
#define MERS_B   0x9D2C5680
#define MERS_C   0xEFC60000

__device__ void RandomInit(uint32 seed);        // re-seed
__device__ void BRandom();                    // output random bits

__device__ __shared__ uint32 mt[MERS_N];                   // state vector

#endif


#define LOWER_MASK ((1LU << MERS_R) - 1)         
#define UPPER_MASK (0xFFFFFFFF << MERS_R)        

__device__ void ori_RandomInit(uint32 seed) 
{
	int i;
	// re-seed generator
	if(threadIdx.x == 0)
	{
		mt[0]= seed & 0xffffffffUL;
		for (i=1; i < MERS_N; i++) 
	{
		mt[i] = (1812433253UL * (mt[i-1] ^ (mt[i-1] >> 30)) + i);
	}
	}
	__syncthreads();
}

__device__ void ori_BRandom() 
{
	// generate 32 random bits
	uint32 y;
	int thdx;

	// block size is 256
	// step 1: 0-226, MERS_N-MERS_M=227
	if (threadIdx.x<MERS_N-MERS_M) 
	{
		y = (mt[threadIdx.x] & UPPER_MASK) | (mt[threadIdx.x+1] & LOWER_MASK);
		y = mt[threadIdx.x+MERS_M] ^ (y >> 1) ^ ( (y & 1)? MERS_A: 0);
	}
	__syncthreads();
	if (threadIdx.x<MERS_N-MERS_M) 
	{
		mt[threadIdx.x] = y;
	}
	__syncthreads();

	// step 2: 227-453
	thdx = threadIdx.x + (MERS_N-MERS_M);
	if (threadIdx.x<MERS_N-MERS_M) 
	{
		y = (mt[thdx] & UPPER_MASK) | (mt[thdx+1] & LOWER_MASK);
		y = mt[threadIdx.x] ^ (y >> 1) ^ ( (y & 1)? MERS_A: 0);
	}
	__syncthreads();
	if (threadIdx.x<MERS_N-MERS_M) 
	{
		mt[thdx] = y;
	}
	__syncthreads();

	// step 3: 454-622
	thdx += (MERS_N-MERS_M);
	if (thdx < MERS_N-1) 
	{
		y = (mt[thdx] & UPPER_MASK) | (mt[thdx+1] & LOWER_MASK);
		y = mt[threadIdx.x+(MERS_N-MERS_M)] ^ (y >> 1) ^ ( (y & 1)? MERS_A: 0);
	}
	__syncthreads();
	if (thdx < MERS_N-1) 
	{
		mt[thdx] = y;
	}
	__syncthreads();

	// step 4: 623
	if (threadIdx.x == 0) 
	{
		y = (mt[MERS_N-1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
		mt[MERS_N-1] = mt[MERS_M-1] ^ (y >> 1) ^ ( (y & 1)? MERS_A: 0);
	}
	__syncthreads();

	// Tempering (May be omitted):
	y ^=  y >> MERS_U;
	y ^= (y << MERS_S) & MERS_B;
	y ^= (y << MERS_T) & MERS_C;
	y ^=  y >> MERS_L;
}


// return a random in [0, max]
// #define Random(max_plus_1)  (BRandom()% (max_plus_1))

__device__ void ptb_RandomInit(uint32 seed, int thread_id_x) 
{
	int i;
	// re-seed generator
	if(thread_id_x == 0)
	{
		mt[0]= seed & 0xffffffffUL;
		for (i=1; i < MERS_N; i++) 
	{
		mt[i] = (1812433253UL * (mt[i-1] ^ (mt[i-1] >> 30)) + i);
	}
	}
	__syncthreads();
}

__device__ void ptb_BRandom(int thread_id_x) 
{
	// generate 32 random bits
	uint32 y;
	int thdx;

	// block size is 256
	// step 1: 0-226, MERS_N-MERS_M=227
	if (thread_id_x<MERS_N-MERS_M) 
	{
		y = (mt[thread_id_x] & UPPER_MASK) | (mt[thread_id_x+1] & LOWER_MASK);
		y = mt[thread_id_x+MERS_M] ^ (y >> 1) ^ ( (y & 1)? MERS_A: 0);
	}
	__syncthreads();
	if (thread_id_x<MERS_N-MERS_M) 
	{
		mt[thread_id_x] = y;
	}
	__syncthreads();

	// step 2: 227-453
	thdx = thread_id_x + (MERS_N-MERS_M);
	if (thread_id_x<MERS_N-MERS_M) 
	{
		y = (mt[thdx] & UPPER_MASK) | (mt[thdx+1] & LOWER_MASK);
		y = mt[thread_id_x] ^ (y >> 1) ^ ( (y & 1)? MERS_A: 0);
	}
	__syncthreads();
	if (thread_id_x<MERS_N-MERS_M) 
	{
		mt[thdx] = y;
	}
	__syncthreads();

	// step 3: 454-622
	thdx += (MERS_N-MERS_M);
	if (thdx < MERS_N-1) 
	{
		y = (mt[thdx] & UPPER_MASK) | (mt[thdx+1] & LOWER_MASK);
		y = mt[thread_id_x+(MERS_N-MERS_M)] ^ (y >> 1) ^ ( (y & 1)? MERS_A: 0);
	}
	__syncthreads();
	if (thdx < MERS_N-1) 
	{
		mt[thdx] = y;
	}
	__syncthreads();

	// step 4: 623
	if (thread_id_x == 0) 
	{
		y = (mt[MERS_N-1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
		mt[MERS_N-1] = mt[MERS_M-1] ^ (y >> 1) ^ ( (y & 1)? MERS_A: 0);
	}
	__syncthreads();

	// Tempering (May be omitted):
	y ^=  y >> MERS_U;
	y ^= (y << MERS_S) & MERS_B;
	y ^= (y << MERS_T) & MERS_C;
	y ^=  y >> MERS_L;
}



__device__ void mix_RandomInit(uint32 seed, int thread_id_x) 
{
	int i;
	// re-seed generator
	if(thread_id_x == 0)
	{
		mt[0]= seed & 0xffffffffUL;
		for (i=1; i < MERS_N; i++) 
	{
		mt[i] = (1812433253UL * (mt[i-1] ^ (mt[i-1] >> 30)) + i);
	}
	}
	asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(128) : "memory");
}

__device__ void mix_BRandom(int thread_id_x) 
{
	// generate 32 random bits
	uint32 y;
	int thdx;

	// block size is 256
	// step 1: 0-226, MERS_N-MERS_M=227
	if (thread_id_x<MERS_N-MERS_M) 
	{
		y = (mt[thread_id_x] & UPPER_MASK) | (mt[thread_id_x+1] & LOWER_MASK);
		y = mt[thread_id_x+MERS_M] ^ (y >> 1) ^ ( (y & 1)? MERS_A: 0);
	}
	asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(128) : "memory");
	if (thread_id_x<MERS_N-MERS_M) 
	{
		mt[thread_id_x] = y;
	}
	asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(128) : "memory");

	// step 2: 227-453
	thdx = thread_id_x + (MERS_N-MERS_M);
	if (thread_id_x<MERS_N-MERS_M) 
	{
		y = (mt[thdx] & UPPER_MASK) | (mt[thdx+1] & LOWER_MASK);
		y = mt[thread_id_x] ^ (y >> 1) ^ ( (y & 1)? MERS_A: 0);
	}
	asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(128) : "memory");
	if (thread_id_x<MERS_N-MERS_M) 
	{
		mt[thdx] = y;
	}
	asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(128) : "memory");

	// step 3: 454-622
	thdx += (MERS_N-MERS_M);
	if (thdx < MERS_N-1) 
	{
		y = (mt[thdx] & UPPER_MASK) | (mt[thdx+1] & LOWER_MASK);
		y = mt[thread_id_x+(MERS_N-MERS_M)] ^ (y >> 1) ^ ( (y & 1)? MERS_A: 0);
	}
	asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(128) : "memory");
	if (thdx < MERS_N-1) 
	{
		mt[thdx] = y;
	}
	asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(128) : "memory");

	// step 4: 623
	if (thread_id_x == 0) 
	{
		y = (mt[MERS_N-1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
		mt[MERS_N-1] = mt[MERS_M-1] ^ (y >> 1) ^ ( (y & 1)? MERS_A: 0);
	}
	asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(128) : "memory");

	// Tempering (May be omitted):
	y ^=  y >> MERS_U;
	y ^= (y << MERS_S) & MERS_B;
	y ^= (y << MERS_T) & MERS_C;
	y ^=  y >> MERS_L;
}

