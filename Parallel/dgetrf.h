#ifndef PARALLEL_DGETRF_H
#define PARALLEL_DGETRF_H

#include <omp.h>
#include <malloc.h>
#include "dgemm.h"

typedef struct pair{
    int key;
    int value;
} PAIR;

typedef struct per{
    int count;
    PAIR* y;
} permutations;

void dgetrf(int N, int M, double *A, int LDA, int *IPIVOT, int *INFO);

#endif
