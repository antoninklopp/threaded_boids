#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>

#define NB_OF_BOIDS 1000
#define NB_BLOCKS 50
#define VISION_DISTANCE 2

__device__ 
float2 get_distance(float2 &position1, float2 &position2){
	if (sqrt(pow(position1.x - position2.x, 2) + pow(position1.y - position2.y, 2))  < VISION_DISTANCE){
		return boid_positions; 
	} 
	else {
		return float2(0, 0); 
	}
}

__device__
void applyCohesionRule(float* boid_positions, float2 &cohesion_vector, int this_index, int size) {

	float2 center_of_mass(0, 0);

	for (int i = 0; i < size; i++)
		if (i != this_index)
			center_of_mass += get_distance(boid_position[this_index], boid_position[i]);
			 
	center_of_mass /= size - 1;

	cohesion_vector = (center_of_mass - boid_position[this_index]) / cohesion_parameter;
}

__global__
void computeCohesion(float* boid_positions, float *new_cohesion, int size){
	int tid = threadIdx.x + blockIdx.x * gridDim.x; // handle the data at this index

	while (tid < NB_OF_BOIDS) {
		float2 cohesion_vector; 
		applyCohesionRule(boid_positions, cohesion_vector, tid, size)
		new_cohesion[tid] = cohesion_vector.x;
		new_cohesion[NB_OF_BOIDS + tid] = cohesion_vector.y;
		tid += blockDim.x * gridDim.x;
	}
}

void cudaComputeCohesion(float *boid_position, float *new_cohesion, int size){
	computeCohesion << <20, 20 >> > (boid_position, new_cohesion, size);
}

__global__ 
void computeNewVelocity(float *old_velocity, float *cohesion, float *separation, float *alignment, float *new_velocity) {
	
	int tid = threadIdx.x + blockIdx.x * gridDim.x; // handle the data at this index

	while (tid < NB_OF_BOIDS) {
	
		new_velocity[tid] = old_velocity[tid] + cohesion[tid] + separation[tid] + alignment[tid];
		new_velocity[NB_OF_BOIDS + tid] = old_velocity[NB_OF_BOIDS + tid] + cohesion[NB_OF_BOIDS + tid] + separation[NB_OF_BOIDS + tid] + alignment[NB_OF_BOIDS + tid];
		tid += blockDim.x * gridDim.x;
	}
}

void cudaComputeNewVelocity(float *old_velocity, float *cohesion, float *separation, float *alignment, float *new_velocity) {

	for (int i = 0; i < 1; i++){
		computeNewVelocity << <1, 1 >> > (old_velocity, cohesion, separation, alignment, new_velocity);
	}
}
