#!/bin/bash

# JobName #
#PBS -N Hybrid

#Which Queue to use #
#PBS -q N10C80

# Max Wall time, Example 1 Minute #
#PBS -l walltime=00:20:00

# How many nodes and tasks per node
#PBS -l select=1:ncpus=1:mpiprocs=4:ompthreads=2:mem=16400000kb

#Change Working directory to SUBMIT directory
cd $PBS_O_WORKDIR

# Run executable #
mpirun --bind-to none hybrid.x < input

