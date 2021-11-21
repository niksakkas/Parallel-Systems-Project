# Parallel-Systems-Projects

The purpose of this work is to develop and evaluate parallel programs in MPI, hybrid MPI + OpenMp
and CUDA that implement the jacobi method (with the variant successive over-relaxation) to solve numerically the
Poisson equation.


### The algorithm of the sequential program (jacobi_serial.c in the sequential folder) works as follows:
1. We have 2 tables: u, u_old. The u_old table is initialized to zero.
2. The new approach of the solution is calculated in the table u for each point (x, y) using the values neighboring points of u_old (5 point stencil) and is implemented in the one_jacobi_iteration function of code.
3. The tables u and u_old are exchanged.
4. Steps 2 and 3 are repeated until the method converges (ie the values after each iteration do not change significantly - error tolerance for the iterrative solver -tol) or until we reach a predefined number of repetitions (mits - maximum solver iterations). In our case, keep the number constant repetitions in mits = 50 and small tol = 1e-13 so that the 50 repetitions are always performed for a correct study escalation.
5. Finally the checkSolution function calculates the error between the numerical and analytical solution.

![Screenshot 2021-11-21 233344](https://user-images.githubusercontent.com/50372934/142779764-a4c544ab-ed51-4607-a50f-08e6df1c9206.png)


### To parallelize this implementation, I do the following:
I partition the u _table into NxN smaller tables. For example, if the u_table is of size 840X840, I can partition it to 10x10 = 100 tables of size 84x84 each.
I then implement the above algorithm on each smaller table. However, I require the neighbouring data (yellow entries) of the tables edges (green entries), in order to complete the 5 point stencil required. Therefore, the processes need to communicate with each other. 

![Screenshot 2021-11-21 233555](https://user-images.githubusercontent.com/50372934/142779833-aea2af1e-a118-4229-a06e-be1540a7c402.png)

### The for loop algorithm that each process implements is:
<ol>
  <li>Send green(neighbouring tables need them) entries, receive yellow</li>
  <li>Calculate inner entries (white).</li>
  <li>Wait until every yellow entry required has been recieved.</li>
  <li>Calculate edges (green).</li>
  <li>Wait until every green entry the neighbours require has been sent.</li>
</ol>

This algorithm is the baseline for all my parallel models.


### My project steps (further described in the Report.pdf file):
<ol>
<li>Introduction</li>
<li>Optimization of the sequential program and preparation for parallelism</li>
<li>Design of MPI parallelism</li>
<li>Optimization of the MPI model</li>
<li>MPI results</li>
<li>Hybrid parallelism design using: MPI + OpenMp</li>
<li>Parallelism using CUDA</li>
<li>Conclusions</li>
</ol>

