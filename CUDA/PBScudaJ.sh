#!/bin/bash

# Which Queue to use, DO NOT CHANGE #
#PBS -q GPUq

# Max Wall time, Example 1 Minute #
#PBS -l walltime=00:05:00

# How many nodes and tasks per node,  1 nodes with 8 tasks(threa$
#PBS -lselect=1:ncpus=1:ompthreads=10:ngpus=1 -lplace=excl

# JobName #
#PBS -N GPUJ

#Change Working directory to SUBMIT directory
cd $PBS_O_WORKDIR

# Run executable #
./cuda < input

# profile executable #
#nvprof ./jacobi_1gpu
