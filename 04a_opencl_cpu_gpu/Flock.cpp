#include "Flock.hpp"
#include "Common.hpp"



using namespace std;

size_t Flock::global_item_size;
size_t Flock::local_item_size;

cl_platform_id Flock::cpu_platform_id;
cl_platform_id Flock::gpu_platform_id;
cl_device_id Flock::cpu_device_id;
cl_device_id Flock::gpu_device_id;
cl_uint Flock::cpu_ret_num_devices;
cl_uint Flock::gpu_ret_num_devices;
cl_uint Flock::cpu_ret_num_platforms;
cl_uint Flock::gpu_ret_num_platforms;

cl_int Flock::ret;
cl_context Flock::cpu_context;
cl_context Flock::gpu_context;
cl_program Flock::cpu_program;
cl_program Flock::gpu_program;

cl_kernel Flock::velocity_kernel;
cl_kernel Flock::position_kernel;
cl_command_queue Flock::cpu_command_queue;
cl_command_queue Flock::gpu_command_queue;

cl_mem Flock::dev_old_velocity;
cl_mem Flock::dev_cohesion;
cl_mem Flock::dev_separation;
cl_mem Flock::dev_alignment;
cl_mem Flock::dev_new_velocity_cpu;
cl_mem Flock::dev_new_velocity_gpu;
cl_mem Flock::dev_old_position;
cl_mem Flock::dev_new_position;

Flock::Flock(Color color) {


	for (int i = 0; i < NB_OF_BOIDS; i++) {

		Boid boid(color, i);

		float x = rand() / (float)RAND_MAX; /* [0, 1.0] */
		x = x * (1 - (-1)) - 1;

		float y = rand() / (float)RAND_MAX; /* [0, 1.0] */
		y = y * (1 - (-1)) - 1;

		boid.position = Vector2D(x, y);
		boids.push_back(boid);


	}

	FILE *fp;
	char fileName[] = "../kernel.cl";
	char *source_str;
	size_t source_size;

	/* Load the source code containing the kernel*/
	fp = fopen(fileName, "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}

	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);

	global_item_size = NB_OF_BOIDS * 2; // Process the entire lists
	local_item_size = 1; // Process in groups of 1

	// Get platform and device information
	cpu_platform_id = NULL;
	cpu_device_id = NULL;

	gpu_platform_id = NULL;
	gpu_device_id = NULL;

	ret = clGetPlatformIDs(1, &cpu_platform_id, &cpu_ret_num_platforms);
	ret = clGetDeviceIDs(cpu_platform_id, CL_DEVICE_TYPE_CPU, 1, &cpu_device_id, &cpu_ret_num_devices);

	ret = clGetPlatformIDs(1, &gpu_platform_id, &gpu_ret_num_platforms);
	ret = clGetDeviceIDs(gpu_platform_id, CL_DEVICE_TYPE_GPU, 1, &gpu_device_id, &gpu_ret_num_devices);

	// Create an OpenCL context
	cpu_context = clCreateContext(NULL, 1, &cpu_device_id, NULL, NULL, &ret);
	gpu_context = clCreateContext(NULL, 1, &gpu_device_id, NULL, NULL, &ret);

	// Create a program from the kernel source
	cpu_program = clCreateProgramWithSource(cpu_context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret); 
	gpu_program = clCreateProgramWithSource(gpu_context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);

	// Build the program
	ret = clBuildProgram(cpu_program, 1, &cpu_device_id, NULL, NULL, NULL);
	ret = clBuildProgram(gpu_program, 1, &gpu_device_id, NULL, NULL, NULL);

	// Create the OpenCL kernel
	velocity_kernel = clCreateKernel(cpu_program, "computeNewVelocity", &ret);
	position_kernel = clCreateKernel(gpu_program, "computeNewPosition", &ret);

	// Create a command queue
	cpu_command_queue = clCreateCommandQueue(cpu_context, cpu_device_id, 0, &ret);
	gpu_command_queue = clCreateCommandQueue(gpu_context, gpu_device_id, 0, &ret);

	// Create memory buffers on the device for each vector 
	dev_old_velocity = clCreateBuffer(cpu_context, CL_MEM_READ_ONLY, NB_OF_BOIDS * 2 * sizeof(float), NULL, NULL);

	dev_cohesion = clCreateBuffer(cpu_context, CL_MEM_READ_ONLY, NB_OF_BOIDS * 2 * sizeof(float), NULL, NULL);
	dev_separation = clCreateBuffer(cpu_context, CL_MEM_READ_ONLY, NB_OF_BOIDS * 2 * sizeof(float), NULL, NULL);
	dev_alignment = clCreateBuffer(cpu_context, CL_MEM_READ_ONLY, NB_OF_BOIDS * 2 * sizeof(float), NULL, NULL);

	dev_new_velocity_cpu = clCreateBuffer(cpu_context, CL_MEM_READ_ONLY, NB_OF_BOIDS * 2 * sizeof(float), NULL, NULL);

	dev_new_velocity_gpu = clCreateBuffer(gpu_context, CL_MEM_READ_ONLY, NB_OF_BOIDS * 2 * sizeof(float), NULL, NULL);

	dev_old_position = clCreateBuffer(gpu_context, CL_MEM_READ_ONLY, NB_OF_BOIDS * 2 * sizeof(float), NULL, NULL);

	dev_new_position = clCreateBuffer(gpu_context, CL_MEM_READ_ONLY, NB_OF_BOIDS * 2 * sizeof(float), NULL, NULL);
}

Vector2D Flock::applyCohesionRule(Boid boid) {

	Vector2D center_of_mass;
	Vector2D cohesion_vector;

	for (int i = 0; i < this->boids.size(); i++)
		if (this->boids[i] != boid)
			if (this->boids[i].position.getDistance(boid.position) < vision_distance)
				center_of_mass += this->boids[i].position;
			 
	center_of_mass /= this->boids.size() - 1;

	cohesion_vector = (center_of_mass - boid.position) / cohesion_parameter;
	return cohesion_vector;
}

Vector2D Flock::applySeparationRule(Boid boid) {

	Vector2D separation_vector;

	for (int i = 0; i < this->boids.size(); i++)
		if (this->boids[i] != boid)
			if (this->boids[i].position.getDistance(boid.position) < separation_distance)
				separation_vector -= (this->boids[i].position - boid.position) / separation_parameter;

	return separation_vector;
}

Vector2D Flock::applyAlignmentRule(Boid boid) {

	Vector2D perceived_velocity;
	Vector2D alignment_vector;

	for (int i = 0; i < this->boids.size(); i++)
		if (this->boids[i] != boid)
			if (this->boids[i].position.getDistance(boid.position) < vision_distance)
				perceived_velocity += this->boids[i].velocity;

	perceived_velocity /= this->boids.size() - 1;

	alignment_vector = (perceived_velocity - boid.velocity) / alignment_parameter;
	return alignment_vector;
}

void Flock::drawBoids() {

	for (int i = 0; i < this->boids.size(); i++) {

		this->boids[i].draw();
	}
}

void Flock::moveBoidsToNewPositions() {

	float old_velocity[NB_OF_BOIDS * 2];

	float cohesion[NB_OF_BOIDS * 2];
	float separation[NB_OF_BOIDS * 2];
	float alignment[NB_OF_BOIDS * 2];

	float new_velocity[NB_OF_BOIDS * 2];

	float old_position[NB_OF_BOIDS * 2];
	float new_position[NB_OF_BOIDS * 2];

	for (int i = 0; i < this->boids.size(); i++) {

		old_position[i] = this->boids[i].position.x;
		old_position[NB_OF_BOIDS + i] = this->boids[i].position.y;

		old_velocity[i] = this->boids[i].velocity.x;
		old_velocity[NB_OF_BOIDS + i] = this->boids[i].velocity.y;

		Vector2D v1 = applyCohesionRule(this->boids[i]);
		Vector2D v2 = applySeparationRule(this->boids[i]);
		Vector2D v3 = applyAlignmentRule(this->boids[i]);

		cohesion[i] = v1.x;
		cohesion[NB_OF_BOIDS + i] = v1.y;

		separation[i] = v2.x;
		separation[NB_OF_BOIDS + i] = v2.y;

		alignment[i] = v3.x;
		alignment[NB_OF_BOIDS + i] = v3.y;
	}

	// New Velocity

	// Enqueue
	ret = clEnqueueWriteBuffer(cpu_command_queue, dev_old_velocity, CL_TRUE, 0, NB_OF_BOIDS * 2 * sizeof(float), old_velocity, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(cpu_command_queue, dev_cohesion, CL_TRUE, 0, NB_OF_BOIDS * 2 * sizeof(float), cohesion, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(cpu_command_queue, dev_separation, CL_TRUE, 0, NB_OF_BOIDS * 2 * sizeof(float), separation, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(cpu_command_queue, dev_alignment, CL_TRUE, 0, NB_OF_BOIDS * 2 * sizeof(float), alignment, 0, NULL, NULL);

	// Set the arguments of the kernel
	ret = clSetKernelArg(velocity_kernel, 0, sizeof(cl_mem), (void *)&dev_old_velocity);
	ret = clSetKernelArg(velocity_kernel, 1, sizeof(cl_mem), (void *)&dev_cohesion);
	ret = clSetKernelArg(velocity_kernel, 2, sizeof(cl_mem), (void *)&dev_separation);
	ret = clSetKernelArg(velocity_kernel, 3, sizeof(cl_mem), (void *)&dev_alignment);

	ret = clSetKernelArg(velocity_kernel, 4, sizeof(cl_mem), (void *)&dev_new_velocity_cpu);

	// Execute the OpenCL kernel on the list
	ret = clEnqueueNDRangeKernel(cpu_command_queue, velocity_kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);

	// Read the memory buffer C on the device to the local variable C
	ret = clEnqueueReadBuffer(cpu_command_queue, dev_new_velocity_cpu, CL_TRUE, 0, NB_OF_BOIDS * 2 * sizeof(float), new_velocity, 0, NULL, NULL);

	// Finish Queue
	ret = clFinish(cpu_command_queue);

	// New position

	// Enqueue
	ret = clEnqueueWriteBuffer(gpu_command_queue, dev_old_position, CL_TRUE, 0, NB_OF_BOIDS * 2 * sizeof(float), old_position, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(gpu_command_queue, dev_new_velocity_gpu, CL_TRUE, 0, NB_OF_BOIDS * 2 * sizeof(float), new_velocity, 0, NULL, NULL);

	// Set the arguments of the kernel
	ret = clSetKernelArg(position_kernel, 0, sizeof(cl_mem), (void *)&dev_old_position);
	ret = clSetKernelArg(position_kernel, 1, sizeof(cl_mem), (void *)&dev_new_velocity_gpu);
	ret = clSetKernelArg(position_kernel, 2, sizeof(cl_mem), (void *)&dev_new_position);

	// Execute the OpenCL kernel on the list
	ret = clEnqueueNDRangeKernel(gpu_command_queue, position_kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);

	// Read the memory buffer C on the device to the local variable C
	ret = clEnqueueReadBuffer(gpu_command_queue, dev_new_position, CL_TRUE, 0, NB_OF_BOIDS * 2 * sizeof(float), new_position, 0, NULL, NULL);

	// Clean up
	ret = clFinish(gpu_command_queue);


	for (int i = 0; i < this->boids.size(); i++) {

		this->boids[i].velocity.x = new_velocity[i];
		this->boids[i].velocity.y = new_velocity[NB_OF_BOIDS + i];
		this->boids[i].position.x = new_position[i];
		this->boids[i].position.y = new_position[NB_OF_BOIDS + i];
	}

	for (int i = 0; i < this->boids.size(); i++) {

		this->boids[i].boundPosition();
		this->boids[i].limitVelocity();
	}
}
