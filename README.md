# Parallel-Systems-Projects

The purpose of this work is to develop and evaluate parallel programs in MPI, hybrid MPI + OpenMp
and CUDA that implement the jacobi method (with the variant successive over-relaxation) to solve numerically the
Poisson equation.


The algorithm works as follows:
1. We have 2 tables: u, u_old. The u_old table is initialized to zero.
2. The new approach of the solution is calculated in the table u for each point (x, y) using the values neighboring points of u_old (5 point stencil) and is implemented in the one_jacobi_iteration function of code.
3. The tables u and u_old are exchanged.
4. Steps 2 and 3 are repeated until the method converges (ie the values after each iteration do not change significantly - error tolerance for the iterrative solver -tol) or until we reach a predefined number of repetitions (mits - maximum solver iterations). In our case, keep the number constant repetitions in mits = 50 and small tol = 1e-13 so that the 50 repetitions are always performed for a correct study escalation.
5. Finally the checkSolution function calculates the error between the numerical and analytical solution


Project steps:

<ol>
<li>Introduction</li>
<li>Optimization of the sequential program and preparation for parallelism</li>
<li>Design of MPI parallelism</li>
<li>Optimization of the MPI model</li>
<li>MPI results</li>
<li>Hybrid parallelism design using: MPI + OpenMp</li>
<li>parallelism using CUDA</li>
<li>Conclusions</li>
</ol>

