#!/bin/bash
for i in $(seq 10 10 200)
    do 
        echo $i; 
        ../bin/04b_opencl_gpu_only/main_gpu $i >> measures.txt; 
    done