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

}

__global__ void streaming(node *grid, node *res, int n, int m) {

}

int main(){
	  ifstream file ("test1.txt");
	  if (file.is_open()) {
			int N, M, size;
			node *grid;
            node *dev_grid;
            
		    file >> N >> M;
            size = N * M * sizeof(node);
            
            grid = (node *)malloc(size);
            cudaMalloc(&dev_grid, size);
            
            for (int i = 0; i < N*M; i++) grid[i] = 0;
            
            unsigned int value;
            node fi_mask = 0b00000001; 
			for (int fi = 0; fi < 4; fi++) {
                cout << std::hex << (unsigned int)fi_mask << ": ";
                for (int i = 0; i < N*M; i++) {
                    file >> value;
                    grid[i] += (value)?fi_mask:0;
                    if (i == 0) cout << value << ":"  << std::hex << (unsigned int)grid[i]  << "\n"; 
                }
                fi_mask = fi_mask << 1;
			}

            /*
			cudaMemcpy(d_grid, grid, size, cudaMemcpyHostToDevice);
			for (int i = 0; i < 1; i++) {
				  collision<<<N, M>>>(d_grid);
				  //streaming<<<N, M>>>(d_grid, d_res, N);
				  //cudaMemcpy(d_grid, d_res, size, cudaMemcpyDeviceToDevice);
			}
			cudaMemcpy(grid, d_grid, size, cudaMemcpyDeviceToHost);

			for (int direc = 0; direc < 4; direc++) {
				  for (int i = 0; i < N*M; i++) {
						cout << grid[i].f[direc] << ' ';
				}
				cout << endl;
			}*/

			free(grid);
			cudaFree(dev_grid);
	  }
	  return 0;
}