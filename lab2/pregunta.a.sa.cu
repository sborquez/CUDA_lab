#include <iostream>
#include <cstdio>
#include <fstream>
using namespace std;

#define BLOCK_SIZE 256

struct Grid {
	  int *f0, *f1, *f2, *f3;
      //right, up, left, down
};

__global__ void collision(Grid *grid, int N, int M){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N*M) {
        if ((!grid->f0[tid] && !grid->f2[tid] && grid->f1[tid] && grid->f3[tid])
            ||(grid->f0[tid] && grid->f2[tid] && !grid->f1[tid] && !grid->f3[tid])) {
            grid->f0[tid] = 1 - grid->f0[tid];
            grid->f1[tid] = 1 - grid->f1[tid];
            grid->f2[tid] = 1 - grid->f2[tid];
            grid->f3[tid] = 1 - grid->f3[tid];
        }
    }
}

__global__ void streaming(Grid *grid, Grid *res, int N, int M) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N*M) {
        int y = tid / M;
        int x = tid % M;

        int tid_right = (x + 1) % M + y * M;
        int tid_left = (x - 1 + M) % M + y * M;
        int tid_above = x + ((y + 1) % N) * M;
        int tid_below = x + ((y - 1 + N) % N) * M;

        res->f0[tid] = grid->f0[tid_left]; //left has right arrow
        res->f2[tid] = grid->f2[tid_right]; //right has left arrow
        res->f3[tid] = grid->f3[tid_above]; //above has down arrow
        res->f1[tid] = grid->f1[tid_below]; //below has up arrow
    }
}

int main(){
    ifstream file ("initial.txt");
    if (!file.is_open()) {
        return 1;
    }
    int N, M, grid_size, array_size;
    Grid *grid;
    Grid *dev_grid, *dev_res;

    file >> N >> M;
    int *f0, *f1, *f2, *f3;
    int *dev_f0, *dev_f1, *dev_f2, *dev_f3;

    grid_size = (int)ceil((float)(N*M)/BLOCK_SIZE);
    array_size = N*M*sizeof(int);

    f0 = (int *)malloc(array_size);
    f1 = (int *)malloc(array_size);
    f2 = (int *)malloc(array_size);
    f3 = (int *)malloc(array_size);
    grid = (Grid *)malloc(sizeof(Grid)); 

    cudaMalloc((void **)&dev_grid, sizeof(Grid));
    cudaMalloc((void **)&dev_res, sizeof(Grid));
    
    cudaMalloc((void **)&dev_f0, array_size);
    cudaMalloc((void **)&dev_f1, array_size);
    cudaMalloc((void **)&dev_f2, array_size);
    cudaMalloc((void **)&dev_f3, array_size);

    for (int i = 0; i < N*M; i++) file >> f0[i];
    for (int i = 0; i < N*M; i++) file >> f1[i];
    for (int i = 0; i < N*M; i++) file >> f2[i];
    for (int i = 0; i < N*M; i++) file >> f3[i];

    cudaMemcpy(dev_f0, f0, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_f1, f1, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_f2, f2, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_f3, f3, array_size, cudaMemcpyHostToDevice);

    grid->f0 = dev_f0;
    grid->f1 = dev_f1;
    grid->f2 = dev_f2;
    grid->f3 = dev_f3;

    cudaMemcpy(dev_grid, grid, sizeof(Grid), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_res, grid, sizeof(Grid), cudaMemcpyHostToDevice);

    int iterations = 1000;
    for (int i = 0; i < iterations; i++) {
        collision<<<grid_size, BLOCK_SIZE>>>(dev_grid, N, M);
        streaming<<<grid_size, BLOCK_SIZE>>>(dev_grid, dev_res, N, M);
        cudaMemcpy(dev_grid, dev_res, sizeof(Grid), cudaMemcpyDeviceToDevice);
    }

    cudaMemcpy(grid, dev_grid, sizeof(Grid), cudaMemcpyDeviceToHost);

    cudaMemcpy(f0, dev_f0, array_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(f1, dev_f1, array_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(f2, dev_f2, array_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(f3, dev_f3, array_size, cudaMemcpyDeviceToHost);

    grid->f0 = f0;
    grid->f1 = f1;
    grid->f2 = f2;
    grid->f3 = f3;

    free(grid);
    free(f0); 
    free(f1); 
    free(f2); 
    free(f3); 
    cudaFree(dev_grid); cudaFree(dev_res);
    cudaFree(dev_f0); cudaFree(dev_f1); cudaFree(dev_f2); cudaFree(dev_f3);
    return 0;
}