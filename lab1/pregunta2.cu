#include <stdio.h>
#include <math.h>
#include <time.h>

#define BLOCK_SIZE 256

// inicializador
void init(float* &y_ji, float* &dev_y_ji, int m) {
    y_ji = (float*)malloc(m*sizeof(float));
    cudaMalloc(&dev_y_ji, m*sizeof(float)); 
    for (int j=0; j<m; ++j) {
        y_ji[j] = j;
    }
    cudaMemcpy(dev_y_ji, y_ji, m*sizeof(float), cudaMemcpyHostToDevice);
}


// Pregunta 2.a
void update_serial(float* y_ji, float dt, float t_i, int m) {
    float t_ip = t_i - dt;  //t_{i-1}
    for(int j=0; j<m; ++j) {
        y_ji[j] += dt*(4*t_ip - y_ji[j] + 3 + j);
    }
}
// Pregunta 2.b
__global__ void update_paralelo(float* dev_y_ji, float dt, float t_i, int m) {
    int Tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (Tid < m) {
        float t_ip = t_i - dt; //t_{i-1}
        dev_y_ji[Tid] += dt*(4*t_ip - dev_y_ji[Tid] + 3 + Tid);
    }
}


/*


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
*/

int main(int argc, int* argv)
{
    int m;
    int n = 1000;
    float dt = 0.001;
    float t_i;
    
    float* y_ji;
    clock_t start, end;
    
    int block_size = BLOCK_SIZE;
    int grid_size;
    float* dev_y_ji;
    float elapsed;

    // Pregunta 2-a
    printf("Pregunta 2.a\n");
    for (int m_=4; m_<=8; m_++) {
        // Preparar variables
        m = (int)powf(10.0, (float)m_);
        t_i = 0.0;
        init(y_ji, dev_y_ji, m);
        // Evaluacion
        start = clock();
        for (int i=0; i<=n; ++i) {
            update_serial(y_ji, dt, t_i, m);
            t_i += dt;
        }
        end = clock();

        //Mostrar resultado 
        printf("m=10^%d elapsed=%f [ms]\n", m_, (double)((1000.0*(end - start))/CLOCKS_PER_SEC));
        
        // Liberar memoria
        cudaFree(dev_y_ji);
        free(y_ji);
    }

    // Pregunta 2-b
    printf("Pregunta 2.b\n");
    for (int m_=4; m_<=8; m_++) {
        // Preparar variables
        m = (int)powf(10.0, (float)m_);
        t_i = 0.0;
        grid_size = (int)ceil((float)(m)/block_size);
        init(y_ji, dev_y_ji, m);
        
        // Evaluacion
        cudaEvent_t ct1, ct2;
        cudaEventCreate(&ct1);
        cudaEventCreate(&ct2);
        cudaEventRecord(ct1);
        for (int i=0; i<=n; ++i) {
            update_paralelo<<<grid_size, block_size>>>(dev_y_ji, dt, t_i, m);
            t_i += dt;
        }
        cudaEventRecord(ct2);
        cudaEventSynchronize(ct2);
        cudaEventElapsedTime(&elapsed, ct1, ct2);
        cudaMemcpy(y_ji, dev_y_ji, (m)*sizeof(float), cudaMemcpyDeviceToHost);

        //Mostrar resultado 
        printf("m=10^%d elapsed=%f [ms]\n", m_, elapsed);
        
        // Liberar memoria
        cudaFree(dev_y_ji);
        free(y_ji);
    }

    // Pregunta 2-c
    printf("Pregunta 2.c\n");
    m = (int)powf(10.0, 8.0);
    for (int block_pow=6; block_pow<=9; block_pow++) {
        // Preparar variables
        t_i = 0.0;
        block_size = (int)powf(2.0, (float)block_pow);
        grid_size = (int)ceil((float)(m)/block_size);
        init(y_ji, dev_y_ji, m);
        
        // Evaluacion
        cudaEvent_t ct1, ct2;
        cudaEventCreate(&ct1);
        cudaEventCreate(&ct2);
        cudaEventRecord(ct1);
        for (int i=0; i<=n; ++i) {
            update_paralelo<<<grid_size, block_size>>>(dev_y_ji, dt, t_i, m);
            t_i += dt;
        }
        cudaEventRecord(ct2);
        cudaEventSynchronize(ct2);
        cudaEventElapsedTime(&elapsed, ct1, ct2);
        cudaMemcpy(y_ji, dev_y_ji, (m)*sizeof(float), cudaMemcpyDeviceToHost);

        //Mostrar resultado 
        printf("block_size=%d elapsed=%f [ms]\n", block_size, elapsed);
        
        // Liberar memoria
        cudaFree(dev_y_ji);
        free(y_ji);
    }
    return 0;
}