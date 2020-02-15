#include <iostream>
#include <cstdio>
#include <fstream>
using namespace std;

#define BLOCK_SIZE 256

struct Node {
	  int f0, f1, f2, f3;
      //right, up, left, down
};

__global__ void collision(Node *grid, int n, int m){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n*m) {
        if ((!grid[tid].f0 && !grid[tid].f2 &&  grid[tid].f1 &&  grid[tid].f3)
            || (grid[tid].f0 &&  grid[tid].f2 && !grid[tid].f1 && !grid[tid].f3))
        {
            grid[tid].f0 = 1 - grid[tid].f0;
            grid[tid].f1 = 1 - grid[tid].f1;
            grid[tid].f2 = 1 - grid[tid].f2;
            grid[tid].f3 = 1 - grid[tid].f3;
        }
    }
}

__global__ void streaming(Node *grid, Node *res, int n, int m) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n*m) {
        int y = tid / m;
        int x = tid % m;
        res[tid].f0 = grid[(x - 1 + m) % m + y * m].f0;    //left has right arrow
        res[tid].f2 = grid[(x + 1) % m + y * m].f2;        //right has left arrow
        res[tid].f3 = grid[ x + ((y + 1) % n) * m].f3;      //above has down arrow
        res[tid].f1 = grid[ x + ((y - 1 + n) % n) * m].f1;  //below has up arrow
    }  
}

int main(){
    ifstream file ("initial.txt");
    if (!file.is_open()) {
        return 1;
    }
    int N, M, size, grid_size;
    Node *grid;
    Node *dev_grid, *dev_res;

    file >> N >> M;
    grid_size = (int)ceil((float)(N*M)/BLOCK_SIZE);

    size = N * M * sizeof(Node);
    grid = (Node *)malloc(size);
    cudaMalloc((void **)&dev_grid, size);
    cudaMalloc((void **)&dev_res, size);

    for (int i = 0; i < N*M; i++) file >> grid[i].f0;
    for (int i = 0; i < N*M; i++) file >> grid[i].f1;
    for (int i = 0; i < N*M; i++) file >> grid[i].f2;
    for (int i = 0; i < N*M; i++) file >> grid[i].f3;

    cudaMemcpy(dev_grid, grid, size, cudaMemcpyHostToDevice);
    int iterations = 1000;
    for (int i = 0; i < iterations; i++) {
        collision<<<grid_size, BLOCK_SIZE>>>(dev_grid, N, M);
        streaming<<<grid_size, BLOCK_SIZE>>>(dev_grid, dev_res, N, M);
        cudaMemcpy(dev_grid, dev_res, size, cudaMemcpyDeviceToDevice);
    }
    cudaMemcpy(grid, dev_grid, size, cudaMemcpyDeviceToHost);

    free(grid);
    cudaFree(dev_grid); cudaFree(dev_res);
    return 0;
}
