#ifndef FLOCK_HPP
#define FLOCK_HPP

#define _CRT_SECURE_NO_DEPRECATE

#include "Boid.hpp"

#include <vector>
#include <thread>

#include <CL/cl.h>
 

#define MAX_SOURCE_SIZE (0x100000)

#define NB_OF_BOIDS 80

class Flock {

	static size_t global_item_size; 
	static size_t local_item_size; 

	static cl_platform_id cpu_platform_id;
	static cl_platform_id gpu_platform_id;
	static cl_device_id cpu_device_id;
	static cl_device_id gpu_device_id;
	static cl_uint cpu_ret_num_devices;
	static cl_uint gpu_ret_num_devices;
	static cl_uint cpu_ret_num_platforms;
	static cl_uint gpu_ret_num_platforms;

	static cl_int ret;
	static cl_context cpu_context;
	static cl_context gpu_context;
	static cl_program cpu_program;
	static cl_program gpu_program;

	static cl_kernel velocity_kernel;
	static cl_kernel position_kernel;
	static cl_command_queue cpu_command_queue;
	static cl_command_queue gpu_command_queue;

	static cl_mem dev_old_velocity;
	static cl_mem dev_cohesion;
	static cl_mem dev_separation;
	static cl_mem dev_alignment;
	static cl_mem dev_new_velocity_cpu;
	static cl_mem dev_new_velocity_gpu;
	static cl_mem dev_old_position;
	static cl_mem dev_new_position;

public:

	Flock(Color color);

	std::vector<Boid> boids;

	Vector2D applyCohesionRule(Boid boid);
	Vector2D applySeparationRule(Boid boid);
	Vector2D applyAlignmentRule(Boid boid);
	
	void drawBoids();

	void moveBoidsToNewPositions();
};

#endif