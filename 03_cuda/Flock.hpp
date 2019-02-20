#ifndef FLOCK_HPP
#define FLOCK_HPP

#include "Boid.hpp"

#include <vector>
#include <thread>

#include <cuda.h>
#include <cuda_runtime.h>

#define NB_OF_BOIDS 1000

class Flock {

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