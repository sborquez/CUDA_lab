#include <iostream>
#include <cstdio>
#include <fstream>
#include <cmath>
#include <ctime>
using namespace std;

#define BLOCK_SIZE 256

typedef unsigned char node;


__global__ void streaming(node *grid, int n, int m)
{
    int tId = threadIdx.x + blockIdx.x * blockDim.x;
    if (tId < n*m) {
        int y = tId / m;
        int x = tId % m;
        node node_ =  grid[(x+1)%m      +             y*m] & 0b01000000 
                    | grid[ x           + ((y-1 + n)%n)*m] & 0b10000000
                    | grid[(x-1 + m)%m  +             y*m] & 0b00010000
                    | grid[ x           +     ((y+1)%n)*m] & 0b00100000;

        grid[x + y*m] = grid[x + y*m] | (node_ >> 4);
    }
}

__global__ void collision(node *grid, int n, int m)
{
    int tId = threadIdx.x + blockIdx.x * blockDim.x;
    if (tId < n*m) {
        int y = tId / m;
        int x = tId % m;
        node node_old = grid[x + y*m] << 4;
        node node_new = grid[x + y*m] << 4;

        int col_borde_sup = (y == 0 && (node_new & 0b00100000));
        int col_borde_inf = (y == n-1 && (node_new & 0b10000000));
        int col_borde_izq = (x == 0 && (node_new & 0b01000000));
        int col_borde_der = (x == m-1 && (node_new & 0b00010000));
        int border = col_borde_sup || col_borde_inf || col_borde_izq || col_borde_der;

        node_new = col_borde_sup * ((node_new & 0b11011111) | 0b10000000) + (1-col_borde_sup) *
            (col_borde_inf * ((node_new & 0b01111111) | 0b00100000) + (1-col_borde_inf) *
                node_new);

        node_new = col_borde_izq * ((node_new & 0b10111111) | 0b00010000) + (1-col_borde_izq) *
            (col_borde_der * ((node_new & 0b11101111) | 0b01000000) + (1-col_borde_der) *
                node_new);

        int col = (node_old == 0b10100000) || (node_old == 0b01010000);
        grid[x + y * m] = (border) * (node_new) + (1-border) *
            (col * (node_old^0b11110000) + (1-col) *
                node_old);
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
        //cout << std::hex << (unsigned int)fi_mask << ": ";
        for (int i = 0; i < N*M; i++) {
            file >> value;
            grid[i] += (value)?fi_mask:0;
            //if (i == 0) cout << value << ":"  << std::hex << (unsigned int)grid[i]  << "\n"; 
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

    free(grid);
    cudaFree(dev_grid);
    return 0;
}
