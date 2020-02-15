#include <stdio.h>
#include <math.h>
#include <time.h>

#define BLOCK_SIZE 256

// Pregunta 1.a
void euler_serial(float* y_t, float dt, int n) {
    for (int i=0; i <= n; ++i) {     
        y_t[i] = 0;
        for (int j=0; j < i; ++j) {
            y_t[i] += 9*powf(j*dt, 2.0) - 4*j*dt + 5;
        }
        y_t[i] = 4 + y_t[i]*dt;
    }
}

// Pregunta 1.b
__global__ void euler_paralelo(float* y_t, float dt, int n) {
    int Tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (Tid <= n) {
        y_t[Tid] = 0;
        for (int j=0; j < Tid; ++j) {
            y_t[Tid] += 9*powf(j*dt, 2.0) - 4*j*dt + 5;
        }
        y_t[Tid] = 4 + y_t[Tid]*dt;
    }
}

// Pregunta 1.c
void euler_hibrida_sumatoria(float dt, int n, float* sumatoria) {
    sumatoria[0] = 0;
    for (int j=1; j <= n; ++j) {
        sumatoria[j] = (9*powf(j*dt, 2.0) - 4*j*dt + 5) + sumatoria[j-1];
    }
}
    
__global__ void euler_hibrida_paralelo(float* y_t, float dt, int n, const float* __restrict__ sumatoria) {
    int Tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (Tid <= n) {
        y_t[Tid] = 4 + sumatoria[Tid]*dt;
    }
}

int main(int argc, int* argv)
{
    int n;
    float dt;
    
    float* y_t;
    float* sumatoria;
    clock_t start, end;
    
    int block_size = BLOCK_SIZE;
    int grid_size;
    float* dev_y_t;
    float* dev_sumatoria;
    float elapsed;

    // Pregunta 1-a
    printf("Pregunta 1.a\n");
    for (int m=1; m<=6; m++) {
        // Preparar variables
        dt = powf(10.0, -1.0*m);
        n = (int)powf(10, m+1);
        y_t = (float*)malloc(sizeof(float)*(n+1));

        // Evaluacion
        start = clock();
        euler_serial(y_t, dt, n);
        end = clock();

        //Mostrar resultado 
        printf("delta_t=10^-%d n=%d elapsed=%f [ms]\n", m, n, (double)((1000.0*(end - start))/CLOCKS_PER_SEC));
        
        free(y_t);
    }

    // Pregunta 1-b
    printf("Pregunta 1.b\n");
    for (int m=1; m<=4 ; m++) {
        // Preparar variables
        dt = powf(10.0, -1.0*m);
        n = (int)powf(10, m+1);
        grid_size = (int)ceil((float)(n+1)/block_size);

        y_t = (float*)malloc(sizeof(float)*(n+1));
        cudaMalloc(&dev_y_t, (n+1)*sizeof(float));

        // Evaluacion
        cudaEvent_t ct1, ct2;
        cudaEventCreate(&ct1);
        cudaEventCreate(&ct2);
        cudaEventRecord(ct1);
        euler_paralelo<<<grid_size, block_size>>>(dev_y_t, dt, n);
        cudaEventRecord(ct2);
        cudaEventSynchronize(ct2);
        cudaEventElapsedTime(&elapsed, ct1, ct2);
        cudaMemcpy(y_t, dev_y_t, (n+1)*sizeof(float), cudaMemcpyDeviceToHost);

        //Mostrar resultado 
        printf("delta_t=10^-%d n=%d elapsed=%f [ms]\n", m, n,elapsed);
        
        cudaFree(dev_y_t);
        free(y_t);
    }

    // Pregunta 1-c
    printf("Pregunta 1.c\n");
    for (int m=1; m<=6 ; m++) {
        // Preparar variables
        dt = powf(10.0, -1.0*m);
        n = (int)powf(10, m+1);
        grid_size = (int)ceil((float)(n+1)/block_size);

        sumatoria = (float*)malloc(sizeof(float)*(n+1));
        y_t = (float*)malloc(sizeof(float)*(n+1));

        cudaMalloc(&dev_y_t, (n+1)*sizeof(float));
        cudaMalloc(&dev_sumatoria, (n+1)*sizeof(float));

        // Sumatoria
        start = clock();
        euler_hibrida_sumatoria(dt, n, sumatoria);

        // Copiar valores
        cudaMemcpy(dev_sumatoria, sumatoria, (n+1)*sizeof(float), cudaMemcpyHostToDevice);
        end = clock();

        // Evaluacion
        cudaEvent_t ct1, ct2;
        cudaEventCreate(&ct1);
        cudaEventCreate(&ct2);
        cudaEventRecord(ct1);
        euler_hibrida_paralelo<<<grid_size, block_size>>>(dev_y_t, dt, n, dev_sumatoria);
        cudaEventRecord(ct2);
        cudaEventSynchronize(ct2);
        cudaEventElapsedTime(&elapsed, ct1, ct2);

        cudaMemcpy(y_t, dev_y_t, (n+1)*sizeof(float), cudaMemcpyDeviceToHost);
        
        //Mostrar resultado 
        printf("delta_t=10^-%d n=%d elapsed=%f [ms]\n", m, n, elapsed + (double)((1000*(end - start))/CLOCKS_PER_SEC));
        
        free(y_t);
        free(sumatoria);

        cudaFree(dev_y_t);
        cudaFree(dev_sumatoria);
    }
    return 0;
}