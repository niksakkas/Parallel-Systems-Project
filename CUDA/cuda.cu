#include <math.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <numeric>
using namespace std;

double checkSolution(double xStart, double yStart,
	int maxXCount, int maxYCount,
	double *u,
	double deltaX, double deltaY,
	double alpha)
{
#define ARRAY(XX,YY) u[(YY)*maxXCount+(XX)]
	int x, y;
	double fX, fY;
	double localError, error = 0.0;
	for (y = 1; y < (maxYCount-1); y++){
		fY = yStart + (y-1)*deltaY;
		for (x = 1; x < (maxXCount-1); x++)
		{
			fX = xStart + (x-1)*deltaX;
			localError = ARRAY(x,y) - (1.0-fX*fX)*(1.0-fY*fY);
			error += localError*localError;
		}
	}
	return sqrt(error)/((maxXCount-2)*(maxYCount-2));
}

__global__ void jacobi(double* u,double* u_old, int n,double alpha, int maxIterationCount,double relax,double*results){
	#define SRC(XX,YY) u[(YY)*(n+2)+(XX)]
	const int tid = threadIdx.x;
	const int blockid = blockIdx.x;
	const int block_length = n/blockDim.x;
	// const int result_index = blockid*blockDim.x + tid;

	const int x_start = tid*block_length + 1;
	const int x_end = (tid+1)*block_length + 1;
	const int y_start = blockid*block_length + 1;
	const int y_end = (blockid+1)*block_length + 1;

	// printf("x: %d, y: %d, x_start: %d, x_end: %d, y_start: %d, y_end: %d, result index: %d\n",tid, blockid,x_start,x_end,y_start,y_end, result_index);
	
	int maxXCount = n+2;
	double xLeft = -1.0, xRight = 1.0;
    double yBottom = -1.0, yUp = 1.0;
    double deltaX = (xRight-xLeft)/(n-1);
    double deltaY = (yUp-yBottom)/(n-1);
	double fX, fY;
    double updateVal;
	double cur_error = 0;
    double cx = 1.0/(deltaX*deltaX);
    double cy = 1.0/(deltaY*deltaY);
    double cc = -2.0*cx-2.0*cy-alpha;
	double* tmp;
	// double total_error=0;

	for (int iterationCount = 0;iterationCount < maxIterationCount; iterationCount++){  
		cur_error = 0; 
		for (int y = y_start; y < y_end; y++){
			fY = yBottom + (y-1)*deltaY;
			for (int x = x_start; x < x_end; x++){
				fX = xLeft + (x-1)*deltaX;
				updateVal = ((SRC(x-1,y) + SRC(x+1,y))*cx + (SRC(x,y-1) + SRC(x,y+1))*cy + SRC(x,y)*cc - (-alpha*(1.0-fX*fX)*(1.0-fY*fY) - 2.0*(1.0-fX*fX) - 2.0*(1.0-fY*fY)))/cc;
				cur_error += updateVal*updateVal;
				u[(y)*maxXCount+(x)] = u_old[(y)*maxXCount+(x)] - relax*updateVal;
				// printf("%f ",SRC(x,y));
				// a++;
			}
		}
	// __syncthreads();	
	tmp = u_old;
	u_old = u;
	u = tmp; 
	//calculate errors :
	// results[result_index] = cur_error;
	// if(tid == 0 && blockid == 0){
	// 	printf("final errors:\n");
	// 	for(int i = 0;i<(blockDim.x)*(blockDim.x);i++){
	// 		printf("%f\n",results[i]);
	// 	}
	// }
	//add the errors
	// if(tid == 0 && blockid == 0){
	// 	for(int i = 0;i<(blockDim.x)*(blockDim.x);i++){
	// 		total_error += results[i];
	// 	}
	// 	printf("total error: %f\n",total_error);
	// 	total_error = sqrt(total_error)/(n*n);
	// 	printf("total error sqrd: %g\n",total_error);
	// 	}
	}

}

int main(){

	int N = 420;
	int n, m, mits;
    double alpha, tol, relax;

    scanf("%d,%d", &n, &m);
    scanf("%lf", &alpha);
    scanf("%lf", &relax);
    scanf("%lf", &tol);
    scanf("%d", &mits);
    printf("-> %d, %d, %g, %g, %g, %d\n", n, m, alpha, relax, tol, mits);

    int allocCount = (n+2)*(m+2);
    double* u = 	(double*)calloc(allocCount, sizeof(double)); //reverse order
    double* u_old = (double*)calloc(allocCount, sizeof(double));
	double* device_u;
	double* device_u_old;
	#define U(XX,YY) u[(YY)*maxCount+(XX)]
	double xLeft = -1.0, xRight = 1.0;
    double yBottom = -1.0, yUp = 1.0;
    double deltaX = (xRight-xLeft)/(n-1);
    double deltaY = (yUp-yBottom)/(n-1);

	const int array_size = allocCount*sizeof(double);
	double* reduce_array = (double*)calloc(N*N, sizeof(double));;
	double* device_reduce_array;

	float elapsed=0;
	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	cudaMalloc(&device_u, array_size);
	cudaMemcpy(device_u, u, array_size, cudaMemcpyHostToDevice);
	cudaMalloc(&device_u_old, array_size);
	cudaMemcpy(device_u_old, u_old, array_size, cudaMemcpyHostToDevice);
	cudaMalloc(&device_reduce_array, N*N);
	cudaMemcpy(device_reduce_array, reduce_array, N*N, cudaMemcpyHostToDevice);

	jacobi <<< N, N >>>(device_u,device_u_old,n,alpha,mits,relax,device_reduce_array);

	cudaMemcpy(u_old, device_u_old, array_size, cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop) ;
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("The elapsed time in gpu was %.2f ms\n", elapsed);

	double absoluteError = checkSolution(xLeft, yBottom,
		n+2, n+2,
		u_old,
		deltaX, deltaY,
		alpha);
	printf("The residual is %g\n", absoluteError);

	cudaFree(device_u);
	cudaFree(device_u_old);
	cudaFree(device_reduce_array);
	return 0;
}
