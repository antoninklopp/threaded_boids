cmake_minimum_required(VERSION 2.8)

#Project's name
project(Boids_GPUa)

add_library(lib_boids SHARED Boid.cpp Vector2D.cpp Flock.cpp)

add_executable(main_cpu_gpu main.cpp)
target_link_libraries(main_cpu_gpu lib_boids ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES} ${OpenCL_LIBRARY})
