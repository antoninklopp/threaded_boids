cmake_minimum_required(VERSION 3.9) 

#Project's name
project(Boids_cpp)

set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")

FIND_PACKAGE(OpenMP REQUIRED)
message(STATUS $ENV{OPENMP_FOUND})
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_subdirectory(glfw)

file(
    GLOB
    source_files
    src/*.cpp
    src/*.h*
)

add_library(boids_cpp SHARED ${source_files})
add_executable(main main.cpp)
target_link_libraries(main boids_cpp ${GLFW_LIBRARIES})