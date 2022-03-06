#include "waxpby.h"
// #include <GL/glut.h>
#include <immintrin.h>
#include <omp.h>
#include <stdio.h>

#include <pthread.h>

#define NUMTHREADS 6

/**
 * @brief Compute the update of a vector with the sum of two scaled vectors
 * 
 * @param n Number of vector elements
 * @param alpha Scalars applied to x
 * @param x Input vector
 * @param beta Scalars applied to y
 * @param y Input vector
 * @param w Output vector
 * @return int 0 if no error
 */
int waxpby (const int n, const double * const x, const double beta, const double * const y, double * const w) {  
  
  // Alpha is always 1, so we don't need branching statements
  // x, y and w are aligned, look in generate_matrix.c, so we dont need to use things like storeu and loadu
  int i = 0;
  int loopN = (n/4)*4;
  __m256d betaVec = _mm256_set1_pd(beta);
  for (i=0; i<loopN; i+=4) {
    __m256d xVec = _mm256_load_pd(x + i);
    __m256d yVec = _mm256_load_pd(y + i);
    _mm256_store_pd(w + i, _mm256_add_pd(xVec, _mm256_mul_pd(betaVec, yVec)));
  }
  for (; i<n; i++) {
    w[i] = x[i] + beta * y[i];
  }

  return 0;

  // static pthread_t tid[NUMTHREADS];
  // // Alpha is always 1, so we don't need branching statements
  // // x, y and w are aligned, look in generate_matrix.c, so we dont need to use things like storeu and loadu
  // int i = 0;
  // int loopN = (n/4)*4;
  // __m256d betaVec = _mm256_set1_pd(beta);
  // for (i=0; i<loopN; i+=4) {
  //   __m256d xVec = _mm256_load_pd(x + i);
  //   __m256d yVec = _mm256_load_pd(y + i);
  //   _mm256_store_pd(w + i, _mm256_add_pd(xVec, _mm256_mul_pd(betaVec, yVec)));
  // }
  // for (; i<n; i++) {
  //   w[i] = x[i] + beta * y[i];
  // }

  // for (int i = 0; i < NUMTHREADS; i++) {
  //   if (pthread_create(&tid[i], NULL, thread_function, NULL)) {
  //       fprintf(stderr, "%s","\nERROR: Could not create thread\n");
  //       fsync(STDERR_FILENO);   // 'fsync()' is used to ensure that the print statement above has finished before exiting
  //       exit(EXIT_FAILURE);
  //   }
  // }

  // return 0;
}

