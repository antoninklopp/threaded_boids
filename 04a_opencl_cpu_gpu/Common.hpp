#ifndef COMMON_HPP
#define COMMON_HPP

#include <GLFW/glfw3.h>

const float pi = 3.14f;

const int width = 1600;
const int height = 900;

// Hyperparameters  - lower = stronger

const float cohesion_parameter = 1.0f;
const float separation_parameter = 1.0f;
const float alignment_parameter = 1.0f;

const float separation_distance = 1.0f;

const float vision_distance = 100.0f;

const float v_lim = 1.0f;

typedef struct Color {

	GLfloat r, g, b;

} Color;

const Color color[8] = { { 0, 0, 0 }, { 0, 0, 1 }, { 0, 1, 0 }, { 0, 1, 1 }, { 1, 0, 0 }, { 1, 0, 1 }, { 1, 1, 0 }, {1, 1, 1} };
const Color colors[24] = { { 0, 0, 0 },{ 0, 0, 0.5 },{ 0, 0, 1 },{ 0, 0.5, 0 },{ 0, 0.5, 0.5 },{ 0, 1, 0 },{ 0, 1, 0.5 },{ 0, 1, 1 },{ 0.5, 0, 0 },{ 0.5, 0, 0.5 },{ 0.5, 0, 1 },{ 0.5, 0.5, 0 },{ 0.5, 0.5, 0.5 },{ 0.5, 1, 0 },{ 0.5, 1, 0.5 },{ 0.5, 1, 1 },{ 1, 0, 0 },{ 1, 0, 0.5 },{ 1, 0, 1 },{ 1, 0.5, 0 },{ 1, 0.5, 0.5 },{ 1, 1, 0 },{ 1, 1, 0.5 },{ 1, 1, 1 } };


#endif
