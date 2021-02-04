#include <stdio.h>
#define N 16
#define BLOCK_SIZE 4
__global__ void transpose(int *input,int *output){
	__shared__ int sharedMemory[BLOCK_SIZE][BLOCK_SIZE + 1];
	//global index
	int indexX = threadIdx.x + blockIdx.x*blockDim.x;
	int indexY = threadIdx.y + blockIdx.y*blockDim.y;
	//transposed index
	int tindexX = threadIdx.x + blockIdx.y*blockDim.x;
	int tindexY = threadIdx.y + blockIdx.x*blockDim.y;
	//local index
	int localIndexX = threadIdx.x;
	int localIndexY = threadIdx.y;
	int index = indexX*N + indexY;
	int transposedIndex = tindexY*N + tindexX;
	sharedMemory[localIndexX][localIndexY] = input[index];
	__syncthreads();
	output[transposedIndex] = sharedMemory[localIndexY][localIndexX];
}	
void fill_data(int *data){
	for(int idx=0;idx < N*N;idx++)
		data[idx] = idx;
}
void print_matrix(int *data,int n){
	for(int i = 0;i < n;i++){
		for(int j = 0;j < n;j++){
			printf("%4d ",data[i*n + j]);
		}
		printf("\n");
	}
}
int main(void){
	int *a,*b;
	int *d_a,*d_b;
	int size = N*N*sizeof(int);
	a = (int*)malloc(size);
	b = (int*)malloc(size);
	fill_data(a);
	cudaMalloc((void**)&d_a,size);
	cudaMalloc((void**)&d_b,size);
	cudaMemcpy(d_a,a,size,cudaMemcpyHostToDevice);
	cudaMemcpy(d_b,b,size,cudaMemcpyHostToDevice);
	dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE,1);
	dim3 gridSize(N/BLOCK_SIZE,N/BLOCK_SIZE,1);
	transpose<<<blockSize,gridSize>>>(d_a,d_b);
	cudaMemcpy(b,d_b,size,cudaMemcpyDeviceToHost);
	printf("Original:\n");
	print_matrix(a,N);
	printf("Transposed:\n");
	print_matrix(b,N);
	free(a);
	free(b);
	cudaFree(d_a);
	cudaFree(d_b);
}

