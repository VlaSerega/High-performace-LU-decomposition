#include "dgemm.h"

inline static void ProdSmall(int m, int n, int k, double alpha, const double *a, int lda, const double *b, int ldb, double beta, double *c, int ldc)
{
    int i, j, p;
    for (i = 0; i < m; ++i)
    {
        double *cp = &c(i, 0);
        register double cura = alpha * a(i, 0);
        //p=0
        for (j = 0; j < n; ++j)
            *(cp++) = cura * b(0, j) + beta * *cp;

        for (p = 1; p < k; p++)
        {
            cp = &c(i, 0);
            register double cura = alpha * a(i, p);
            const double *tmpb = &b(p, 0);
            for (j = 0; j + 4 < n; j += 4)
            {
                *(cp++) += cura * *(tmpb++);
                *(cp++) += cura * *(tmpb++);
                *(cp++) += cura * *(tmpb++);
                *(cp++) += cura * *(tmpb++);
            }
            for (; j < n; ++j)
                *(cp++) += cura * *(tmpb++);
        }
    }
}


void dgemm(int m, int n, int k, double alpha, const double *a, int lda, const double *b, int ldb, double beta, double *c, int ldc)
{
    int i, j, p, tmpi, tmpp;
    int mm, nn, kk;

#pragma omp parallel for private(i, j, p, tmpi, tmpp, mm, nn, kk)
    for (i = 0; i < m; i += CM)
    {
        for (p = 0; p < k; p += CK)
        {
            int tmp = 0;
            mm = min(CM, m - i);
            kk = min(CK, k - p);
            double *new_a = (double *) malloc(mm * kk * sizeof(double));

            for (tmpi = 0; tmpi < mm; ++tmpi)
            {
                const double *tmpa = &a(i + tmpi, p);
                for (tmpp = 0; tmpp < kk; ++tmpp)
                    new_a[tmp++] = *(tmpa++);
            }

            for (j = 0; j < n; j += CN)
            {
                nn = min(CN, n - j);
                if (p == 0)
                {
                    ProdSmall(mm, nn, kk, alpha, new_a, kk, &b(p, j), ldb, beta, &c(i, j), ldc);
                } else {
                    ProdSmall(mm, nn, kk, alpha, new_a, kk, &b(p, j), ldb, 1.0, &c(i, j), ldc);
                }
            }
            free(new_a);

        }
    }

}