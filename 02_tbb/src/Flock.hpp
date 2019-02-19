#ifndef FLOCK_HPP
#define FLOCK_HPP

#include "Boid.hpp"

#include <vector>
#include <thread>

#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"	

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