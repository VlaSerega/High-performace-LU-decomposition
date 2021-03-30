#include "dgetrf.h"

void swapRows(const int M, const int ida, double *const A, permutations steps) {
    register int count = steps.count;

    for (int rows = 0; rows < count; ++rows) {
        register double * tmpAk = A + steps.y[rows].key * ida;
        register double * tmpAv = A + steps.y[rows].value * ida;

        for (int columns = 0; columns < M; ++columns) {
            double tmp = *tmpAk;
            *tmpAk = *tmpAv;
            *tmpAv = tmp;
            tmpAk ++;
            tmpAv ++;
        }
    }

}

void findU12(const int countRows, const int countColumns, double *const A, const int ida) {
    int startElementU12 = countRows;

#pragma omp parallel
    {
        int start = (countColumns / omp_get_num_threads()) * omp_get_thread_num() + (omp_get_thread_num() < (countColumns % omp_get_num_threads()) ? omp_get_thread_num() : countColumns % omp_get_num_threads());
        int count = (countColumns / omp_get_num_threads()) + (omp_get_thread_num() < (countColumns % omp_get_num_threads()));

        for (int rows = 0; rows < countRows; ++rows) {
            for (int k = 0; k < rows; ++k) {
                register double tmp = A[rows * ida + k];
                register double *tmpAr =  A + startElementU12 + rows * ida + start;
                register double *tmpAk =  A + startElementU12 + k * ida + start;

                for (int columns = 0; columns < count; ++columns) {
                    *tmpAr -= tmp * *tmpAk;
                    tmpAk ++;
                    tmpAr ++;
                }
            }
        }
    }
}

void changeA22(int N, int M, double* A, int ida) {

    int countRows = N - (M / 2);
    int countColumns = M - (M / 2);
    int startElementL21 = (ida * (M / 2));
    int startElementU12 = M / 2;
    int startElementA22 = (M / 2) * ida + (M / 2);

#pragma omp parallel for
    for (int rows = 0; rows < countRows; ++rows) {        //Mul L21 * U12
        for (int k = 0; k < M /2; ++k) {
            double tmp = A[startElementL21 + ida * rows + k];
            for (int columns = 0; columns < countColumns; ++columns) {
                A[startElementA22 + rows * ida + columns] -= tmp * A[startElementU12 + k * ida + columns];
            }
        }
    }
}

void changePermutation(permutations *steps1,permutations *steps2, int M) {

    int count2 = steps2->count;
    int count1 = steps1->count;

    for (int rows = 0; rows < count2; ++rows) {
        (*steps1).y[rows + count1].key = (*steps2).y[rows].key + (M / 2);
        (*steps1).y[rows + count1].value = (*steps2).y[rows].value + (M / 2);
    }
    steps1->count += count2;
}

void normdgetrf(int N, int M, int ida, double* A, permutations *steps) {

    permutations tmpsteps;  //tmpsteps = P2, steps = P1

    if (M == 1) { //Matrix has only one column
        int flag = -1;

        for (int rows = 0; rows < N; ++rows) {
            if (A[rows * ida] != 0) {   //Try to find non zero number
                flag = rows;
                break;
            }
        }

        if (flag != -1) {

            if (flag != 0) {
                (*steps).y[(*steps).count].key = 0;            //Change permutation row
                (*steps).y[(*steps).count].value = flag;
                (steps->count) ++;

                swapRows(M, ida, A, *steps);
            }

            for (int rows = 1; rows < N; ++rows) {
                A[rows * ida] = A[rows * ida] / A[0];   //Save U and L matrix in A matrix
            }
        }

        return; //Finish function
    }

    normdgetrf(N, M / 2, ida, A, steps);

    swapRows(M - (M / 2), ida, A + (M / 2), *steps);     //Equivalently P1 *| A12 |
    //       							               | A22 |

    findU12(M / 2, M - (M / 2), A, ida);

    dgemm((M / 2), N - (M / 2), N - (M / 2), -1.0, A + ida * (M / 2), ida, A + (M / 2), ida, 1.0, A + (M / 2) + ida * (M / 2), ida);

    changeA22(N, M, A, ida);

    tmpsteps.y = (PAIR*)malloc(sizeof(PAIR) * (N - (M / 2)));
    tmpsteps.count = 0;

    normdgetrf(N - (M / 2), M - (M / 2), ida, A + (ida + 1) * (M / 2), &tmpsteps);    //Send left down part (A22)

    swapRows(M / 2, ida, A + (M / 2) * ida, tmpsteps);    //Equivalently P2 * L21

    changePermutation(steps, &tmpsteps, M);    //Mul P1 * P2
}

void dgetrf(int N, int M, double *A, int LDA, int *IPIVOT, int *INFO) { //Main algorithm function
    permutations steps;
    int count;

    steps.count = 0;

    if (N < 1 || M < 1 || A == NULL || LDA < M) {
        *INFO = -1;
        return;
    }

    steps.y = (PAIR*)malloc(sizeof(PAIR) * M);

    for (int i = 0; i < N; ++i) {
        IPIVOT[i] = i;
    }

    if (N >= M) {

        normdgetrf(N, M, LDA, A, &steps);

    }
    else {

        normdgetrf(N, N, LDA, A, &steps);

        findU12(N, M - N, A, LDA);

    }

    count = steps.count;

    for (int i = count - 1; i >= 0; --i) {
        IPIVOT[steps.y[i].key] ^= IPIVOT[steps.y[i].value];
        IPIVOT[steps.y[i].value] ^= IPIVOT[steps.y[i].key];
        IPIVOT[steps.y[i].key] ^= IPIVOT[steps.y[i].value];
    }

    *INFO = 0;
}