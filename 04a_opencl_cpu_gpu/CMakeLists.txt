cmake_minimum_required(VERSION 2.8)

#Project's name
project(Boids_GPUa)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
include_directories(library ${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS})

add_subdirectory(glfw)

include_directories(glfw/include)

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})

add_library(lib_boids SHARED Boid.cpp Vector2D.cpp Flock.cpp)

add_executable(main main.cpp)
target_link_libraries(main lib_boids ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES} ${OpenCL_LIBRARY})