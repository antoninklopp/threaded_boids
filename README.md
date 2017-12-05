## boids_basic_threading

Boids Simulation using:

* C++ Threading
* Intel's Threading Building Blocks (TBB)
* Nvidia's Compute Unified Device Architecture (CUDA)
* Apple's Open Computing Language (OpenCL)

## Required packages

* OpenGL
* OpenGL Utility Toolkit (GLUT)
* OpenGL Extension Wrangler (GLEW)
* TBB
* CUDA
* OpenCL

## Outputs

~ 60 fps  
CPU : i7-4790k  
GPU : GTX 760  

**C++ Threading** - 7 flocks, 20 boids each (140 boids in total)
![cpp_threading.gif](../outputs/cpp_threading.gif)

**TBB** - 7 flocks, 50 boids each (350 boids in total)
![tbb.gif](outputs/tbb.gif)

**CUDA** - 7 flocks, 100 boids each (700 boids in total)
![cuda.gif](outputs/cuda.gif)

**OpenCL** - 24 flocks, 80 boids each (1920 boids in total)
![opencl.gif](outputs/opencl.gif)
