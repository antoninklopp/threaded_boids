cmake_minimum_required(VERSION 2.8)

#Project's name
project(Boids_GPUb)

add_library(lib_boids_gpub SHARED Boid.cpp Vector2D.cpp Flock.cpp)

add_executable(main_gpu main.cpp)
target_link_libraries(main_gpu lib_boids_gpub ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES} ${OpenCL_LIBRARY})