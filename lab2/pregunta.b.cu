#include <stdio.h>
#include <math.h>
#include <time.h>

#define BLOCK_SIZE 256

// Implementacion 3
#include <iostream>
#include <cstdio>
#include <fstream>
using namespace std;

typedef unsigned char node;

__global__ void collision(node *grid, int n, int m)
{
    int tId = threadIdx.x + blockIdx.x * blockDim.x;
    if (tId < n*m) {
        int y = tId / m;
        int x = tId % m;
        // Esto (<< 4) es para evitar las condiciones de carrera en streaming
        // al utilizar la mitad del byte como un buffer
        node node_ = grid[x + y*m] << 4;
        grid[x + y * m] = ((node_ == 0b10100000) || (node_ == 0b01010000))? (node_^0b11110000):node_;
    }
}

__global__ void streaming(node *grid, int n, int m) {
    int tId = threadIdx.x + blockIdx.x * blockDim.x;
    if (tId < n*m) {
        int y = tId / m;
        int x = tId % m;
        node node_ = (grid[(x+1)%m      +             y*m] & 0b01000000)
                   | (grid[ x           + ((y-1 + n)%n)*m] & 0b10000000)
                   | (grid[(x-1 + m)%m  +             y*m] & 0b00010000)
                   | (grid[ x           +     ((y+1)%n)*m] & 0b00100000);

        grid[x + y*m] = grid[x + y*m] | (node_ >> 4);
    }
}


int main(){
	ifstream file ("initial.txt");
	if (!file.is_open()) {
        return 1;
    }
    int N, M, size;
    file >> N >> M;
    size = N * M * sizeof(node);
    
    int block_size = BLOCK_SIZE;
    int grid_size = (int)ceil((float)(N*M)/block_size);
    
    node *grid;
    node *dev_grid;
    grid = (node*)malloc(size);
    cudaMalloc(&dev_grid, size);
    
    for (int i = 0; i < N*M; i++) grid[i] = 0;
    
    unsigned int value;
    node fi_mask = 0b00000001; 
    for (int fi = 0; fi < 4; fi++) {
        for (int i = 0; i < N*M; i++) {
            file >> value;
            grid[i] += (value)?fi_mask:0;
        }
        fi_mask = fi_mask << 1;
    }
    cudaMemcpy(dev_grid, grid, size, cudaMemcpyHostToDevice);

    int iterations = 1000;
    for (int i = 0; i < iterations; i++) {
        collision<<<grid_size, block_size>>>(dev_grid, N, M);
        streaming<<<grid_size, block_size>>>(dev_grid, N, M);
    }
    cudaMemcpy(grid, dev_grid, size, cudaMemcpyDeviceToHost);

    /*
    fi_mask = 0b00000001; 
    for (int fi = 0; fi < 4; fi++) {
        cout << "f_" << fi << ": ";
        for (int i = 0; i < N*M; i++) {
            //cout << (bool)(grid[i] & fi_mask) << " ";
            cout << ((grid[i] >> fi)%2) << " ";
            //cout << (unsigned int)grid[i] << " ";
        }
        fi_mask = fi_mask << 1;
        cout << "\n";
    }*/

    free(grid);
    cudaFree(dev_grid);
    return 0;
}