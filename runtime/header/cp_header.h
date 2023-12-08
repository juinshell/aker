#define BLOCKSIZEX 16
#define BLOCKSIZEY 8
#define UNROLLX 8

#define MAXATOMS 64
// #define MAXATOMS 4000
// #define VOLSIZEX 1024
// #define VOLSIZEY 512
#define VOLSIZEX 4096
#define VOLSIZEY 4096
// #define ATOMCOUNT 40000
#define ATOMCOUNT 4000

#define CUERR { cudaError_t err; \
  if ((err = cudaGetLastError()) != cudaSuccess) { \
  printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__); \
  return -1; }}

// Max constant buffer size is 64KB, minus whatever
// the CUDA runtime and compiler are using that we don't know about.
// At 16 bytes for atom, for this program 4070 atoms is about the max
// we can store in the constant buffer.
__constant__ float4 atominfo[MAXATOMS];

#include "pets_common.h"
#define CP_GRID_DIM (SM_NUM * 2)

// This function copies atoms from the CPU to the GPU and
// precalculates (z^2) for each atom.
int copyatomstoconstbuf(float *atoms, int count, float zplane) {
	if (count > MAXATOMS) {
		printf("Atom count exceeds constant buffer storage capacity\n");
		return -1;
	}

	float atompre[4*MAXATOMS];
	int i;
	for (i=0; i<count*4; i+=4) {
		atompre[i    ] = atoms[i    ];
		atompre[i + 1] = atoms[i + 1];
		float dz = zplane - atoms[i + 2];
		atompre[i + 2]  = dz*dz;
		atompre[i + 3] = atoms[i + 3];
	}

	cudaMemcpyToSymbol(atominfo, atompre, count * 4 * sizeof(float), 0);
	CUERR // check and clear any existing errors

	return 0;
}


/* initatoms()
 * Store a pseudorandom arrangement of point charges in *atombuf.
 */
static int initatoms(float **atombuf, int count, dim3 volsize, float gridspacing) {
	dim3 size;
	int i;
	float *atoms;

	srand(54321);			// Ensure that atom placement is repeatable

	atoms = (float *) malloc(count * 4 * sizeof(float));
	*atombuf = atoms;

	// compute grid dimensions in angstroms
	size.x = gridspacing * volsize.x;
	size.y = gridspacing * volsize.y;
	size.z = gridspacing * volsize.z;

	for (i=0; i<count; i++) {
		int addr = i * 4;
		atoms[addr    ] = (rand() / (float) RAND_MAX) * size.x; 
		atoms[addr + 1] = (rand() / (float) RAND_MAX) * size.y; 
		atoms[addr + 2] = (rand() / (float) RAND_MAX) * size.z; 
		atoms[addr + 3] = ((rand() / (float) RAND_MAX) * 2.0) - 1.0;  // charge
	}  

	return 0;
}