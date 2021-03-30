#ifndef PARALLEL_DGEMM_H
#define PARALLEL_DGEMM_H

#include <malloc.h>

#define CM 128
#define CN 256
#define CK 128


#define a(i, j)     a[(i) * lda + (j) ]
#define b(i, j)     b[(i) * ldb + (j) ]
#define c(i, j)     c[(i) * ldc + (j) ]

#define min(i, j) ((i) < (j) ? (i) : (j))

void dgemm(int m, int n, int k, double alpha, const double *a, int lda, const double *b, int ldb, double beta, double *c, int ldc);

#endif