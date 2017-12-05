#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdio.h>

#define NB_OF_BOIDS 100

__global__ void computeNewVelocity(float *old_velocity, float *cohesion, float *separation, float *alignment, float *new_velocity) {
	
	int tid = blockIdx.x; // handle the data at this index

	if (tid < NB_OF_BOIDS) {
	
		new_velocity[tid] = old_velocity[tid] + cohesion[tid] + separation[tid] + alignment[tid];
		new_velocity[NB_OF_BOIDS + tid] = old_velocity[NB_OF_BOIDS + tid] + cohesion[NB_OF_BOIDS + tid] + separation[NB_OF_BOIDS + tid] + alignment[NB_OF_BOIDS + tid];
	}	
}

void cudaComputeNewVelocity(float *old_velocity, float *cohesion, float *separation, float *alignment, float *new_velocity) {

	computeNewVelocity << <NB_OF_BOIDS, 1 >> > (old_velocity, cohesion, separation, alignment, new_velocity);
}
