#ifndef BOID_HPP
#define BOID_HPP

#include <vector>
#include "Vector2D.hpp"
#include "Common.hpp"

using namespace std;

class Boid {

private:

	Color color;
	int id;

public:

	Boid(Color color, int id);

	Vector2D velocity;
	Vector2D position;

	bool operator==(Boid boid);
	bool operator!=(Boid boid);

	void Boid::computeNewPosition();

	void boundPosition();
	void limitVelocity();

	void draw();
};

#endif