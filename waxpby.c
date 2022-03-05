#include "waxpby.h"
// #include <GL/glut.h>
#include <immintrin.h>
#include <omp.h>
#include <stdio.h>

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
int waxpby (const int n, const double alpha, const double * const x, const double beta, const double * const y, double * const w) {  
  int i = 0;
  int loopN = (n/4)*4;

  if (alpha==1.0) {
    __m128d betaVec = _mm_set1_pd(beta);
    // #pragma omp parallel for         // WHY DOES THIS PRAGMA NOT WORK???   TRY ADDING CRITICAL/ATOMIC??
    for (i=0; i<loopN; i+=2) {
      __m128d xVec = _mm_load_pd(x + i);
      __m128d yVec = _mm_load_pd(y + i);
      _mm_store_pd(w + i, _mm_add_pd(xVec, _mm_mul_pd(betaVec, yVec)));     // MAYBE REARRANGE
    }
    for (; i<n; i++) {
      w[i] = x[i] + beta * y[i];
    }
  } 
  else if(beta==1.0) {
    __m256d alphaVec = _mm256_set1_pd(alpha);
    #pragma omp parallel for
    for (i=0; i<loopN; i+=4) {
      __m256d xVec = _mm256_load_pd(x + i);
      __m256d yVec = _mm256_load_pd(y + i);
      _mm256_store_pd(w + i, _mm256_add_pd(_mm256_mul_pd(alphaVec, xVec), yVec));
    }
    for (; i<n; i++) {
      w[i] = alpha * x[i] + y[i];
    }
  } 
  else {
    __m256d betaVec = _mm256_set1_pd(beta);
    __m256d alphaVec = _mm256_set1_pd(alpha);
    #pragma omp parallel for
    for (i=0; i<loopN; i+=4) {
      __m256d xVec = _mm256_load_pd(x + i);
      __m256d yVec = _mm256_load_pd(y + i);
      _mm256_store_pd(w + i, _mm256_add_pd(_mm256_mul_pd(alphaVec, xVec), _mm256_mul_pd(betaVec, yVec)));
    }
    for (; i<n; i++) {
      w[i] = alpha * x[i] + beta * y[i];
    }
  }

  return 0;
}