#include "Flock.hpp"
#include "Common.hpp"

using namespace std;

extern void cudaComputeNewVelocity(float *old_velocity, float *cohesion, float *separation, float *alignment, float *new_velocity);
extern void cudaComputeCohesion(float* boid_positions, float* velocity, float *new_velocity, int size); 

Flock::Flock(Color color) {

	for (int i = 0; i < NB_OF_BOIDS; i++) {

		Boid boid(color);

		float x = rand() / (float)RAND_MAX; /* [0, 1.0] */
		x = x * (1 - (-1)) - 1;

		float y = rand() / (float)RAND_MAX; /* [0, 1.0] */
		y = y * (1 - (-1)) - 1;

		boid.position = Vector2D(x, y);
		boids.push_back(boid);
	}
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

	float *dev_old_velocity, *dev_new_velocity;

	float *dev_cohesion, *dev_separation, *dev_alignment, *dev_position;

	float old_velocity[NB_OF_BOIDS * 2];
	float new_velocity[NB_OF_BOIDS * 2];

	float cohesion[NB_OF_BOIDS * 2];
	float separation[NB_OF_BOIDS * 2];
	float alignment[NB_OF_BOIDS * 2];

	float position[NB_OF_BOIDS * 2];

	cudaMalloc((void**)&dev_old_velocity, NB_OF_BOIDS * 2 * sizeof(float));
	cudaMalloc((void**)&dev_new_velocity, NB_OF_BOIDS * 2 * sizeof(float));

	cudaMalloc((void**)&dev_cohesion, NB_OF_BOIDS * 2 * sizeof(float));
	cudaMalloc((void**)&dev_separation, NB_OF_BOIDS * 2 * sizeof(float));
	cudaMalloc((void**)&dev_alignment, NB_OF_BOIDS * 2 * sizeof(float));

	cudaMalloc((void**)&dev_position, NB_OF_BOIDS * 2 * sizeof(float));

	for(int i = 0; i < this->boids.size(); i++) {

		old_velocity[i] = this->boids[i].velocity.x;
		old_velocity[NB_OF_BOIDS + i] = this->boids[i].velocity.y;

		// Vector2D v1 = applyCohesionRule(this->boids[i]);
		// Vector2D v2 = applySeparationRule(this->boids[i]);
		// Vector2D v3 = applyAlignmentRule(this->boids[i]);

		// cohesion[i] = v1.x;
		// cohesion[NB_OF_BOIDS + i] = v1.y;

		// separation[i] = v2.x;
		// separation[NB_OF_BOIDS + i] = v2.y;

		// alignment[i] = v3.x;
		// alignment[NB_OF_BOIDS + i] = v3.y;

		position[i] = this->boids[i].position.x;
		position[i + NB_OF_BOIDS] = this->boids[i].position.y;  
	}

	cudaMemcpy(dev_position, position, NB_OF_BOIDS * 2 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_old_velocity, old_velocity, NB_OF_BOIDS * 2 * sizeof(float), cudaMemcpyHostToDevice);

	cudaComputeCohesion(dev_position, dev_old_velocity, dev_new_velocity, NB_OF_BOIDS);

	cudaMemcpy(new_velocity, dev_new_velocity, NB_OF_BOIDS * 2 * sizeof(float), cudaMemcpyDeviceToHost);


	// cudaMemcpy(dev_cohesion, cohesion, NB_OF_BOIDS * 2 * sizeof(float), cudaMemcpyHostToDevice);
	// cudaMemcpy(dev_separation, separation, NB_OF_BOIDS * 2 * sizeof(float), cudaMemcpyHostToDevice);
	// cudaMemcpy(dev_alignment, alignment, NB_OF_BOIDS * 2 * sizeof(float), cudaMemcpyHostToDevice);

	// cudaComputeNewVelocity(dev_old_velocity, dev_cohesion, dev_separation, dev_alignment, dev_new_velocity);

	// cudaMemcpy(new_velocity, dev_new_velocity, NB_OF_BOIDS * 2 * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(dev_old_velocity);
	cudaFree(dev_new_velocity);

	cudaFree(dev_cohesion);
	cudaFree(dev_separation);
	cudaFree(dev_alignment);
	
	cudaDeviceSynchronize();

	for (int i = 0; i < this->boids.size(); i++) {

		this->boids[i].velocity.x = new_velocity[i];
		this->boids[i].velocity.y = new_velocity[NB_OF_BOIDS + i];
	}

	for (int i = 0; i < this->boids.size(); i++) {

		this->boids[i].computeNewPosition();
		this->boids[i].boundPosition();
		this->boids[i].limitVelocity();
	}
}
