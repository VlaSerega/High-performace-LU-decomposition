#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <malloc.h>
#include "dgetrf.h"

double* randGenerate(int n, int m) { // Create random matrix

    double* matr = (double *) malloc(sizeof(double) * n * m);

    srand(time(NULL));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            matr[i * m + j] = (double)rand();
        }
    }

    matr[0] = 0;
    matr[1] = 0;
    matr[2] = 1;
    matr[3] = 1;
    matr[4] = 1;
    matr[5] = 1;
    matr[6] = 1;
    matr[7] = 2;
    matr[8] = 1;


    return matr;
}

int main(int argc, char **argv) {
    int N = atoi(argv[2]);
    int M = atoi(argv[3]);

    double *matr =  randGenerate(N, M);
    double *LU = (double *)malloc(sizeof(double) * M * N);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            LU[i * M + j] = matr[i * M + j];
        }
    }

    int* IPIVIOT = (int*)malloc(sizeof(int) * N);
    int INFO;

    int NUM_THREADS = atoi(argv[1]);

    omp_set_num_threads(NUM_THREADS);

    dgetrf(N, M, LU, M, IPIVIOT, &INFO);

    free(IPIVIOT);
    free(matr);
}
