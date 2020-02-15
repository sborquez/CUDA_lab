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


__global__ void collision_and_streaming(node *grid, int n, int m, int iteration)
{
    int tId = threadIdx.x + blockIdx.x * blockDim.x;
    if (tId < n*m) {
        int y = tId / m;
        int x = tId % m;

        // Usamos el valor iteration para determinar en que mitad del byte operar.
        // iteration par -> leer 00001111 y escribir 11110000
        // iteration impar -> leer 11110000 y escribir 00001111
        unsigned char read  = (iteration % 2)? 0b11110000:0b00001111; 
        
        // leemos las celdas de las cuatro direcciones
        // consideramos los bordes, si tiene una flecha ortogonal, entonces 
        // le damos un nodo que corresponda, en el streaming, a la refleccion
        // estos nodos "reflejados"

        // Si el vecino esta fuera del rango, simplemente no tiene particulas
        node right= (x == (m-1))?0b00000000:(grid[(x+1) +     y*m] & read);
        node up   = (y == 0)    ?0b00000000:(grid[x     + (y-1)*m] & read);
        node left = (x == 0)    ?0b00000000:(grid[(x-1) + y*m    ] & read);
        node down = (y == (n-1))?0b00000000:(grid[x     + (y+1)*m] & read);

        // Si el vecino esta en el borde, entonces crea una reflexion si tengo una particula ortogonal
        // en otro caso solo es el nodo vecino y podemos revisar si colisiona
        // solo hacemos la reflexion en la direccion que le influye al nodo actual.
        // las colisiones solo ocurren fuera de los bordes

        // tipos de colisiones validas
        unsigned char collision1 = 0b10101010 & read;
        unsigned char collision2 = 0b01010101 & read;
        right = ((x+1) == (m-1))? ((right & (0b00010001&read))? (0b010001000&read) : right)
                                : ((y != 0 && y != (n-1) && (right == collision1 || right == collision2)) ? right^read: right);
        up    = ((y-1) == 0)    ? ((up & (0b00100010&read))? (0b10001000&read)     : up) 
                                : ((x != 0 && x != (m-1) && (up && collision1    || up == collision2))    ? up^read   : up); 
        left  = ((x-1) == 0)    ? ((left & (0b01000100&read))? (0b00010001&read)   : left)
                                : ((y != 0 && y != (n-1) && (left == collision1  || left == collision2))  ? left^read : left);
        down = ((y+1) == (n-1)) ? ((down & (0b10001000&read))? (0b00100010&read)   : down)
                                : ((x != 0 && x != (m-1) && (down == collision1  || down == collision2))  ? down^read : down);

        // unir resultados
        node node_new = 
              (right & 0b01000100)
            | (up    & 0b10001000)
            | (left  & 0b00010001)
            | (down  & 0b00100010);

        // el resultado se escribre en 4 bit para escritura
        node_new = (iteration % 2)?(node_new >> 4): (node_new << 4); 
        
        // las unimos y guardamos el resultado en el lado del byte correspondiente
        grid[x + y*m] = (grid[x + y*m] & read) | node_new;
    }
}

int main(){
	ifstream file ("initial.txt");
	//ifstream file ("test1.txt");
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
    //int iterations = 2;
    for (int i = 0; i < iterations; i++) {
        collision_and_streaming<<<grid_size, block_size>>>(dev_grid, N, M, i);
    }
    cudaMemcpy(grid, dev_grid, size, cudaMemcpyDeviceToHost);

    
    /*
    fi_mask = 0b0001000; 
    for (int fi = 4*(iterations%2); fi < 4 + 4*(iterations%2); fi++) {
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