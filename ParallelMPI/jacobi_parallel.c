/************************************************************
 * Program to solve a finite difference
 * discretization of the screened Poisson equation:
 * (d2/dx2)u + (d2/dy2)u - alpha u = f
 * with zero Dirichlet boundary condition using the iterative
 * Jacobi method with overrelaxation.
 *
 * RHS (source) function
 *   f(x,y) = -alpha*(1-x^2)(1-y^2)-2*[(1-x^2)+(1-y^2)]
 *
 * Analytical solution to the PDE
 *   u(x,y) = (1-x^2)(1-y^2)
 *
 * Current Version: Christian Iwainsky, RWTH Aachen University
 * MPI C Version: Christian Terboven, RWTH Aachen University, 2006
 * MPI Fortran Version: Dieter an Mey, RWTH Aachen University, 1999 - 2005
 * Modified: Sanjiv Shah,        Kuck and Associates, Inc. (KAI), 1998
 * Author:   Joseph Robicheaux,  Kuck and Associates, Inc. (KAI), 1998
 *
 * Unless READ_INPUT is defined, a meaningful input dataset is used (CT).
 *
 * Input : n     - grid dimension in x direction
 *         m     - grid dimension in y direction
 *         alpha - constant (always greater than 0.0)
 *         tol   - error tolerance for the iterative solver
 *         relax - Successice Overrelaxation parameter
 *         mits  - maximum iterations for the iterative solver
 *
 * On output
 *       : u(n,m)       - Dependent variable (solution)
 *       : f(n,m,alpha) - Right hand side function
 *
 *************************************************************/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>

/*************************************************************
 * Performs one iteration of the Jacobi method and computes
 * the residual value.
 *
 * NOTE: u(0,*), u(maxXCount-1,*), u(*,0) and u(*,maxYCount-1)
 * are BOUNDARIES and therefore not part of the solution.
 *************************************************************/
// #define SRC(XX,YY) src[(YY)*maxXCount+(XX)]
// #define DST(XX,YY) dst[(YY)*maxXCount+(XX)]


/**********************************************************
 * Checks the error between numerical and exact solutions
 **********************************************************/
double checkSolution(double xStart, double yStart,
                     int maxXCount, int maxYCount,
                     double *u,
                     double deltaX, double deltaY,
                     double alpha)
{
#define U(XX,YY) u[(YY)*maxXCount+(XX)]
    int x, y;
    double fX, fY;
    double localError, error = 0.0;

    for (y = 1; y < (maxYCount-1); y++)
    {
        fY = yStart + (y-1)*deltaY;
        for (x = 1; x < (maxXCount-1); x++)
        {
            fX = xStart + (x-1)*deltaX;
            localError = U(x,y) - (1.0-fX*fX)*(1.0-fY*fY);
            error += localError*localError;
        }
    }
    return sqrt(error)/((maxXCount-2)*(maxYCount-2));
}


int main(int argc, char **argv)
{

    int n, m, mits;
    double alpha, tol, relax;
    double maxAcceptableError;
    double error;
    double total_error = INFINITY;
    double *u, *u_old, *tmp;
    int allocCount;
    int iterationCount, maxIterationCount;
    double t1, t2;

    int x, y;
    double fX, fY;
    double cur_error = 0.0;
    double updateVal;
    double f;

    //read file stuff
    int bytes_read;
    char* part;
    FILE * fp;
    char * line = NULL;
    size_t len = 0;
    ssize_t read;

    fp = fopen("input", "r");
    if (fp == NULL)
        exit(EXIT_FAILURE);

    bytes_read = getline(&line, &len, fp);
    part = strtok(line,",");
    n = atoi(part);
    part = strtok(line,"\n");
    m = atoi(part);

    // n=6;
    // m=6;

    bytes_read = getline(&line, &len, fp);
    alpha = strtod(part,NULL);
    bytes_read = getline(&line, &len, fp);
    relax = strtod(part,NULL);
    bytes_read = getline(&line, &len, fp);
    tol = strtod(part,NULL);
    bytes_read = getline(&line, &len, fp);
    mits = atoi(part);
    fclose(fp);
    free(line);

    printf("-> %d, %d, %g, %g, %g, %d\n", n, m, alpha, relax, tol, mits);

    
    maxIterationCount = mits;
    maxAcceptableError = tol;

    // Solve in [-1, 1] x [-1, 1]
    double xLeft = -1.0, xRight = 1.0;
    double yBottom = -1.0, yUp = 1.0;
    
    int myid, numprocs; 
    MPI_Status s_status, r_status; 

    MPI_Request r_north_request;  
    MPI_Request r_south_request;  
    MPI_Request r_west_request;  
    MPI_Request r_east_request;  
    MPI_Request s_north_request; 
    MPI_Request s_south_request; 
    MPI_Request s_west_request; 
    MPI_Request s_east_request; 

    int tag = 23; /* arbitrary value */ 
    //position variables (will change to 0 if neighbour doesnt exist)
    int north = 1;
    int south = 1;
    int west = 1;
    int east = 1;

    MPI_Init(NULL,NULL);   

    t1 = MPI_Wtime();
    
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);   
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);   
    //process variables
    int N = sqrt(numprocs);
    int block_x = myid/N;
    int block_y = myid%N;

    int block_length = n/N;   //length of each block row
    int block_height = m/N;   //height of each block column

    //save neighbour ids (so that we dont need to calculate them over and over)
    int north_neightbour_id = myid-N;
    int south_neightbour_id = myid+N;
    int west_neightbour_id = myid-1;
    int east_neightbour_id = myid+1;

    //incoming halo buffers
    double* south_v = (double*)calloc(block_length, sizeof(double));
    double* north_v = (double*)calloc(block_length, sizeof(double));
    double* west_v = (double*)calloc(block_height, sizeof(double));
    double* east_v = (double*)calloc(block_height, sizeof(double));

    allocCount = (block_length+2)*(block_height+2);
    // Those two calls also zero the boundary elements
    u = 	(double*)calloc(allocCount, sizeof(double)); //reverse order
    u_old = (double*)calloc(allocCount, sizeof(double));
    #define SRC(XX,YY) u_old[(YY)*maxXCount+(XX)]
    #define DST(XX,YY) u[(YY)*maxXCount+(XX)]
    
    if (u == NULL || u_old == NULL){
        printf("Not enough memory for two %ix%i matrices\n", block_length, block_height);
        exit(1);
    }
    double deltaX = (xRight-xLeft)/(n-1);
    double deltaY = (yUp-yBottom)/(m-1);
    iterationCount = 0;
    error = HUGE_VAL;
    clock_t start = clock(), diff;
    // Coefficients
    double cx = 1.0/(deltaX*deltaX);
    double cy = 1.0/(deltaY*deltaY);
    double cc = -2.0*cx-2.0*cy-alpha;

    int maxXCount = block_length + 2;
    int maxYCount = block_height + 2;
    
    MPI_Datatype column;
    MPI_Type_vector(block_height, 1, block_length, MPI_DOUBLE, &column);
    MPI_Type_commit(&column);

    MPI_Datatype row;
    MPI_Type_contiguous(block_length, MPI_DOUBLE, &row);
    MPI_Type_commit(&row);
    
    if (block_x == 0){
        north = 0;
    }
    else if(block_x == N-1){
        south = 0;
    }
    if (block_y == 0){
        west = 0;
    }
    else if (block_y == N-1){
        east = 0;
    }

    MPI_Barrier(MPI_COMM_WORLD);
//send and receive
    for (int iterationCount = 0;iterationCount < maxIterationCount; iterationCount++){    	

        if (north != 0){
            MPI_Irecv(north_v, 1, row, north_neightbour_id, tag, MPI_COMM_WORLD, &r_south_request);
            MPI_Isend(&SRC(1,1), 1, row, north_neightbour_id, tag, MPI_COMM_WORLD, &s_south_request);        
        }
        if (south != 0){
            MPI_Irecv(south_v, 1, row, south_neightbour_id, tag, MPI_COMM_WORLD, &r_north_request);
            MPI_Isend(&SRC(1,block_height), 1, row, south_neightbour_id, tag, MPI_COMM_WORLD, &s_north_request);
        }
        if (west != 0){
            MPI_Irecv(west_v, 1, row, west_neightbour_id, tag, MPI_COMM_WORLD, &r_east_request);
            MPI_Isend(&SRC(1,1), 1, column, west_neightbour_id, tag, MPI_COMM_WORLD, &s_east_request);
        }
        if (east != 0){
            MPI_Irecv(east_v, 1, row, east_neightbour_id, tag, MPI_COMM_WORLD, &r_west_request);
            MPI_Isend(&SRC(block_length,1), 1, column, east_neightbour_id, tag, MPI_COMM_WORLD, &s_west_request);
        }
//calculate white

        cur_error = 0;
        // printf("maxcount: %d\n",maxXCount);
        for (y = 2; y < (maxYCount-2); y++){
            fY = yBottom + (y-1)*deltaY;
            for (x = 2; x < (maxXCount-2); x++){
                fX = xLeft + (x-1)*deltaX;
                updateVal = ((SRC(x-1,y) + SRC(x+1,y))*cx + (SRC(x,y-1) + SRC(x,y+1))*cy + SRC(x,y)*cc - (-alpha*(1.0-fX*fX)*(1.0-fY*fY) - 2.0*(1.0-fX*fX) - 2.0*(1.0-fY*fY)))/cc;
                // printf("updateval: %f\n",updateVal);
                cur_error += updateVal*updateVal;
                u[(y)*maxXCount+(x)] = u_old[(y)*maxXCount+(x)] - relax*updateVal;
                // printf("update by: %f\n",relax*updateVal);
            }
        }    
//wait receive
        if (north != 0){
            MPI_Wait(&r_south_request, &r_status);
        }
        if (south != 0){
            MPI_Wait(&r_north_request, &r_status); 
        }
        if (west != 0){
            MPI_Wait(&r_east_request, &r_status);
        }
        if (east != 0){
            MPI_Wait(&r_west_request, &r_status);
        }

//calculate green
    //calculate green lines and columns
        //calculate north line using north halo   (y=1)
        fY = yBottom;
        for (x = 2; x < maxXCount - 2; x++){
            fX = xLeft + (x-1)*deltaX;
            updateVal = ((SRC(x-1,1) + SRC(x+1,1))*cx + (north_v[x-1] + SRC(x,2))*cy + SRC(x,1)*cc - (-alpha*(1.0-fX*fX)*(1.0-fY*fY) - 2.0*(1.0-fX*fX) - 2.0*(1.0-fY*fY)))/cc;

            cur_error += updateVal*updateVal;
            u[maxXCount+(x)] = u_old[maxXCount+(x)] - relax*updateVal;
        }
        //calculate south line using south halo   (y=maxYCount-2)
        fY = yBottom + (maxYCount-3)*deltaY;
        for (x = 2; x < (maxXCount-2); x++){
            fX = xLeft + (x-1)*deltaX;
            updateVal = ((SRC(x-1,maxYCount-2) + SRC(x+1,maxYCount-2))*cx + (SRC(x,maxYCount-3) + east_v[x-1])*cy + SRC(x,maxYCount-2)*cc - (-alpha*(1.0-fX*fX)*(1.0-fY*fY) - 2.0*(1.0-fX*fX) - 2.0*(1.0-fY*fY)))/cc;
            cur_error += updateVal*updateVal;
            u[(maxYCount-2)*maxXCount+(x)] = u_old[(maxYCount-2)*maxXCount+(x)] - relax*updateVal;
        }
        //calculate west line using west halo
        fX = xLeft;
        for (y = 2; y < (maxYCount-2); y++){
            fY = yBottom + (y-1)*deltaY;
            updateVal = ((west_v[y-1] + SRC(2,y))*cx + (SRC(1,y-1) + SRC(1,y+1))*cy + SRC(1,y)*cc - (-alpha*(1.0-fX*fX)*(1.0-fY*fY) - 2.0*(1.0-fX*fX) - 2.0*(1.0-fY*fY)))/cc;
            cur_error += updateVal*updateVal;
            u[(y)*maxXCount+1] = u_old[(y)*maxXCount+1] - relax*updateVal;
        }
        //calculate east line using east halo
        fX = xLeft + (maxXCount-3)*deltaX;
        for (y = 2; y < (maxYCount-2); y++){
            fY = yBottom + (y-1)*deltaY;
            updateVal = ((SRC(maxXCount-3,y) + east_v[y-1])*cx + (SRC(maxXCount-2,y-1) + SRC(maxXCount-2,y+1))*cy + SRC(maxXCount-2,y)*cc - (-alpha*(1.0-fX*fX)*(1.0-fY*fY) - 2.0*(1.0-fX*fX) - 2.0*(1.0-fY*fY)))/cc;
            cur_error += updateVal*updateVal;
            u[(y)*maxYCount+maxXCount-2] = u_old[(y)*maxYCount+maxXCount-2] - relax*updateVal;
        }
    //calculate green corners
        //calculate top left corner
        fY = yBottom ;
        fX = xLeft ;
        updateVal = ((west_v[0] + SRC(2,1))*cx + (north_v[0] + SRC(1,2))*cy + SRC(1,1)*cc - (-alpha*(1.0-fX*fX)*(1.0-fY*fY) - 2.0*(1.0-fX*fX) - 2.0*(1.0-fY*fY)))/cc;
        cur_error += updateVal*updateVal;
        u[maxXCount+1] = u_old[maxXCount+1] - relax*updateVal;
        //calculate top right corner
        fY = yBottom ;
        fX = xLeft + (maxXCount-3)*deltaX;
        updateVal = ((SRC(maxXCount-3,1) + east_v[0])*cx + (north_v[maxXCount-1] + SRC(maxXCount-2,2))*cy + SRC(maxXCount-2,1)*cc - (-alpha*(1.0-fX*fX)*(1.0-fY*fY) - 2.0*(1.0-fX*fX) - 2.0*(1.0-fY*fY)))/cc;
        cur_error += updateVal*updateVal;
        u[maxXCount+maxXCount-2] = u_old[maxXCount+maxXCount-2] - relax*updateVal;
        //calculate bottom left corner
        fY = yBottom + (maxYCount-3)*deltaY;
        fX = xLeft;
        updateVal = (( west_v[maxYCount-1] + SRC(2,maxYCount-2))*cx + (south_v[0] + SRC(1,maxYCount-3))*cy + SRC(1,maxYCount-2)*cc - (-alpha*(1.0-fX*fX)*(1.0-fY*fY) - 2.0*(1.0-fX*fX) - 2.0*(1.0-fY*fY)))/cc;
        cur_error += updateVal*updateVal;
        u[(maxYCount-2)*maxXCount+1] = u_old[(maxYCount-2)*maxXCount+1] - relax*updateVal;
        //calculate bottom right corner
        fY = yBottom + (maxYCount-3)*deltaY;
        fX = xLeft + (maxXCount-3)*deltaX;
        updateVal = (( east_v[maxYCount-1] + SRC(maxXCount-3,maxYCount-2))*cx + (south_v[maxYCount-1] + SRC(maxXCount-2,maxYCount-3))*cy + SRC(maxXCount-2,maxYCount-2)*cc - (-alpha*(1.0-fX*fX)*(1.0-fY*fY) - 2.0*(1.0-fX*fX) - 2.0*(1.0-fY*fY)))/cc;
        cur_error += updateVal*updateVal;
        u[(maxYCount-2)*maxXCount+(maxXCount-2)] = u_old[(maxYCount-2)*maxXCount+(maxXCount-2)] - relax*updateVal;
    //wait send
        if (north != 0){
            MPI_Wait(&s_south_request, &s_status); 
        }
        if (south != 0){
            MPI_Wait(&s_north_request, &s_status);
        }
        if (west != 0){
            MPI_Wait(&s_east_request, &s_status);
        }
        if (east != 0){
            MPI_Wait(&s_west_request, &s_status);
        }   
        tmp = u_old;
        u_old = u;
        u = tmp; 

        MPI_Allreduce(&cur_error, &error, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        // if (myid==0)
        // {
        //     total_error = sqrt(error)/((n+2)*(m+2));
        //     printf("%d\n",iterationCount);
        //     printf("total error: %g\n",total_error);            
        // }
    }
    // if (myid==0){
    //         //u_old
    //         printf("u_old:\n");
    //         for (y = 1; y < (maxYCount-1); y++){
    //             for (x = 1; x < (maxXCount-1); x++){
    //                 printf("%f ",u_old[(y)*maxXCount+(x)]);
    //             }
    //             printf("\n");
    //         }
    //         //u
    //         printf("u:\n");
    //         for (y = 1; y < (maxYCount-1); y++){
    //             for (x = 1; x < (maxXCount-1); x++){
    //                 printf("%f ",u[(y)*maxXCount+(x)]);
    //             }
    //             printf("\n");
    //         }    
    //     }

    t2 = MPI_Wtime();
    
    if (myid==0){
        diff = clock() - start;
        int msec = diff * 1000 / CLOCKS_PER_SEC;
        printf("Time taken %d seconds %d milliseconds\n", msec/1000, msec%1000);
        printf("Residual %g\n",error);

    // u_old holds the solution after the most recent buffers swap
    
    // double absoluteError = checkSolution(xLeft, yBottom,
    //                                      n+2, m+2,
    //                                      u_old,
    //                                      deltaX, deltaY,
    //                                      alpha);
    // printf("The error of the iterative solution is %g\n", absoluteError);
    }
    MPI_Finalize();
    return 0;
}

/*
mpicc -O3 jacobi_parallel.c -o jacobi_parallel.x -lm
mpicc -O3 -g jacobi_parallel.c -L/opt/mpiP-3.5/lib -lmpiP -lbfd -lunwind -o jacobi_parallel.x -lm

ssh -i /home/niksakkas/argo-rbs-key argo291@argo-rbs.cloud.iasa.gr

scp -i /home/niksakkas/argo-rbs-key jacobi_parallel.c argo291@argo-rbs.cloud.iasa.gr:/home/pool/argo291
*/
