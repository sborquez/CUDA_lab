#include <iostream>
#include <cstdio>
#include <fstream>
using namespace std;

#define BLOCK_SIZE 256

__global__ void kernelA(int *A, int *x, int *b, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N*N) {
        int i = tid / N;
        int j = tid % N;
        atomicAdd(&(b[i]), A[N*i+j] * x[j]);
    }
}

__global__ void kernelx(int *A, int *x, int *b, int N) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    if (j < N){
        for (int i = 0; i < N; i++) {
            atomicAdd(&(b[i]), A[N*i+j] * x[j]);
        }
    }
}

__global__ void kernelb(int *A, int *x, int *b, int N){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N){
        for (int j = 0; j < N; j++) {
            b[i] += A[N*i+j] * x[j];
        }
    }
}

__global__ void KernelRed(int *A, int *x, int *b, int N)
{
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int k = threadIdx.x;
    __shared__ int mult[BLOCK_SIZE];
    mult[k] = 0;
    if (j < N){
        for (int i = 0; i < N; i++) {
            //guardar resultado del thread
            mult[k] = A[N*i+j] * x[j];
            __syncthreads();
            // reducimos el resultado parcial de bloque (12-MemoriaComparida.pdf 28/42)
            for (int s = BLOCK_SIZE/2; s >= 1; s /= 2) {
                //if (k < s) mult[k] = mult[k] + mult[k+s];
                mult[k] = (k < s)? mult[k] + mult[k+s]: mult[k];
                __syncthreads();
            }
            if (k == 0) atomicAdd(&(b[i]), mult[0]);
            //atomicAdd(&(b[i]), (k==0)?mult[0]:0);
        }
    }
}

void setup(int *A, int *x, int *b, int N) {
    for (int i=0; i<N; ++i) {
        x[i] = 1;
        b[i] = 0;
        for (int j = 0; j < N; ++j) {
            A[N*i + j] = 1;
        }
    }
}

bool check(int *b, int N) {
    for (int i = 0; i < N; i++) {
        //printf("i=%d\t%d\n", i, b[i]);
        if (b[i] != 10000) 
            return false;
    }
    return true;
}

int main(){
    int N = 10000;
    int *A, *b, *x;
    int *dev_A, *dev_b, *dev_x;
    int grid_size;
    int block_size = BLOCK_SIZE;
    float elapsed;

    // Mallocs
    A = (int*)malloc(sizeof(int)*(N*N));
    b = (int*)malloc(sizeof(int)*(N));
    x = (int*)malloc(sizeof(int)*(N));

    cudaMalloc(&dev_A, (N*N)*sizeof(int));
    cudaMalloc(&dev_b, (N)*sizeof(int));
    cudaMalloc(&dev_x, (N)*sizeof(int));
    
    /*
    KernelA
    */
    {
        // Setup KernelA
        setup(A, x, b, N);
        cudaMemcpy(dev_A, A, N * N * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_x, x, N * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);
        grid_size = (int)ceil((float)(N*N)/block_size);
        
        // Cuda events
        cudaEvent_t ct1, ct2;
        cudaEventCreate(&ct1);
        cudaEventCreate(&ct2);
        cudaEventRecord(ct1);
        kernelA<<<grid_size, block_size>>>(dev_A, dev_x, dev_b, N);
        cudaEventRecord(ct2);
        cudaEventSynchronize(ct2);
        cudaEventElapsedTime(&elapsed, ct1, ct2);

        // Check
        cudaMemcpy(b, dev_b, N * sizeof(int), cudaMemcpyDeviceToHost);        
        if (check(b, N)) printf("KernelA is correct.\n");
        else             printf("KernelA is incorrect.\n");        
        printf("elapsed=%f [ms]\n", elapsed);
    }

    /*
    Kernelx
    */
    {
        // Setup Kernelx
        setup(A, x, b, N);
        cudaMemcpy(dev_A, A, N * N * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_x, x, N * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);
        grid_size = (int)ceil((float)(N)/block_size);
        
        // Cuda events
        cudaEvent_t ct1, ct2;
        cudaEventCreate(&ct1);
        cudaEventCreate(&ct2);
        cudaEventRecord(ct1);
        kernelx<<<grid_size, block_size>>>(dev_A, dev_x, dev_b, N);
        cudaEventRecord(ct2);
        cudaEventSynchronize(ct2);
        cudaEventElapsedTime(&elapsed, ct1, ct2);

        // Check
        cudaMemcpy(b, dev_b, N * sizeof(int), cudaMemcpyDeviceToHost);        
        if (check(b, N)) printf("Kernelx is correct.\n");
        else             printf("Kernelx is incorrect.\n");        
        printf("elapsed=%f [ms]\n", elapsed);
    }

    /*
    Kernelb
    */
    {
        // Setup Kernelb
        setup(A, x, b, N);
        cudaMemcpy(dev_A, A, N * N * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_x, x, N * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);
        grid_size = (int)ceil((float)(N)/block_size);
        
        // Cuda events
        cudaEvent_t ct1, ct2;
        cudaEventCreate(&ct1);
        cudaEventCreate(&ct2);
        cudaEventRecord(ct1);
        kernelb<<<grid_size, block_size>>>(dev_A, dev_x, dev_b, N);
        cudaEventRecord(ct2);
        cudaEventSynchronize(ct2);
        cudaEventElapsedTime(&elapsed, ct1, ct2);

        // Check
        cudaMemcpy(b, dev_b, N * sizeof(int), cudaMemcpyDeviceToHost);        
        if (check(b, N)) printf("Kernelb is correct.\n");
        else             printf("Kernelb is incorrect.\n");        
        printf("elapsed=%f [ms]\n", elapsed);
    }

    /*
    KernelRed
    */
    {
        // Setup Kernelb
        setup(A, x, b, N);
        cudaMemcpy(dev_A, A, N * N * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_x, x, N * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);
        grid_size = (int)ceil((float)(N)/block_size);
        
        // Cuda events
        cudaEvent_t ct1, ct2;
        cudaEventCreate(&ct1);
        cudaEventCreate(&ct2);
        cudaEventRecord(ct1);
        KernelRed<<<grid_size, block_size>>>(dev_A, dev_x, dev_b, N);
        cudaEventRecord(ct2);
        cudaEventSynchronize(ct2);
        cudaEventElapsedTime(&elapsed, ct1, ct2);
    
        // Check
        cudaMemcpy(b, dev_b, N * sizeof(int), cudaMemcpyDeviceToHost);        
        if (check(b, N)) printf("KernelRed is correct.\n");
        else             printf("KernelRed is incorrect.\n");        
        printf("elapsed=%f [ms]\n", elapsed);
    }

    // Free memory
    free(A); free(b); free(x);
    cudaFree(dev_A); cudaFree(dev_b); cudaFree(dev_x);
    return 0;
}
