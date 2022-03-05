#include "waxpby.h"
// #include <GL/glut.h>
#include <immintrin.h>
#include <omp.h>

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
  // Since we are dealing with doubles, our loop factor has to be two in order to use intrinsics: 128 / 64 = 2
  omp_set_num_threads(6);
  int i = 0;
  int loopN = (n/2)*2;

  if (alpha==1.0) {
    __m128d betaVec = _mm_set1_pd(beta);
    // #pragma omp parallel for         // WHY DOES THIS PRAGMA NOT WORK???   TRY ADDING CRITICAL/ATOMIC??
    for (i=0; i<loopN; i+=2) {
      __m128d xVec = _mm_load_pd(x + i);
      __m128d yVec = _mm_load_pd(y + i);
      _mm_store_pd(w + i, _mm_add_pd(_mm_mul_pd(betaVec, yVec), xVec));     // MAYBE REARRANGE
    }
    for (; i<n; i++) {
      w[i] = x[i] + beta * y[i];
    }
  } 
  else if(beta==1.0) {
    __m128d alphaVec = _mm_set1_pd(alpha);
    #pragma omp parallel for
    for (i=0; i<loopN; i+=2) {
      __m128d xVec = _mm_load_pd(x + i);
      __m128d yVec = _mm_load_pd(y + i);
      _mm_store_pd(w + i, _mm_add_pd(_mm_mul_pd(alphaVec, xVec), yVec));
    }
    for (; i<n; i++) {
      w[i] = alpha * x[i] + y[i];
    }
  } 
  else {
    __m128d betaVec = _mm_set1_pd(beta);
    __m128d alphaVec = _mm_set1_pd(alpha);
    #pragma omp parallel for
    for (i=0; i<loopN; i+=2) {
      __m128d xVec = _mm_load_pd(x + i);
      __m128d yVec = _mm_load_pd(y + i);
      _mm_store_pd(w + i, _mm_add_pd(_mm_mul_pd(alphaVec, xVec), _mm_mul_pd(betaVec, yVec)));
    }
    for (; i<n; i++) {
      w[i] = alpha * x[i] + beta * y[i];
    }
  }

  return 0;
}

/******************** ORIGINAL ********************/ 
// #include "waxpby.h"

// /**
//  * @brief Compute the update of a vector with the sum of two scaled vectors
//  * 
//  * @param n Number of vector elements
//  * @param alpha Scalars applied to x
//  * @param x Input vector
//  * @param beta Scalars applied to y
//  * @param y Input vector
//  * @param w Output vector
//  * @return int 0 if no error
//  */
// int waxpby (const int n, const double alpha, const double * const x, const double beta, const double * const y, double * const w) {  
//   if (alpha==1.0) {
//     for (int i=0; i<n; i++) {
//       w[i] = x[i] + beta * y[i];
//     }
//   } else if(beta==1.0) {
//     for (int i=0; i<n; i++) {
//       w[i] = alpha * x[i] + y[i];
//     }
//   } else {
//     for (int i=0; i<n; i++) {
//       w[i] = alpha * x[i] + beta * y[i];
//     }
//   }

//   return 0;
// }

// // 15.04 seconds
