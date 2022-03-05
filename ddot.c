#include "ddot.h"
#include <immintrin.h>
#include <omp.h>
#include <stdio.h>

/**
 * @brief Compute the dot product of two vectors
 * 
 * @param n Number of vector elements
 * @param x Input vector
 * @param y Input vector
 * @param result Pointer to scalar result value
 * @return int 0 if no error
 */
int ddot (const int n, const double * const x, const double * const y, double * const result) {  
  
  // // THIS IS MORE ACCURATE AND QUICKER??? WHY???
  double local_result = 0.0;
  // #pragma omp parallel for reduction(+: local_result)
  // for (int i=0; i<n; i++) {
  //   local_result += x[i]*y[i];
  // }

  // THIS IS FASTEST but NOT AS ACCURATE??
  if (x == y) {
    #pragma omp parallel for reduction(+: local_result)
    for (int i=0; i<n; i++) {
      local_result += x[i]*x[i];
    }
  }
  else {
    #pragma omp parallel for reduction(+: local_result)
    for (int i=0; i<n; i++) {
      local_result += x[i]*y[i];
    }
  }

  *result = local_result;
  return 0;

  // double * local_result = (double *) _mm_malloc(sizeof(double), 64);     // maybe 32?? Size of cache line
  // local_result[0] = 0.0;
  // int i = 0;
  // int loopN = (n/2)*2;
  // __m128d lrVec = _mm_set1_pd(local_result[0]);
  // #pragma omp parallel for reduction(+: local_result[0])
  // for (i = 0; i < loopN; i+=2) {
  //   __m128d xVec = _mm_load_pd(x + i);
  //   __m128d yVec = _mm_load_pd(y + i);
  //   lrVec = _mm_add_pd(lrVec, _mm_mul_pd(xVec, yVec));
  // }
  // _mm_store_pd(local_result + 0, lrVec);
  // for (; i < n; i++) {
  //   local_result[0] += x[i]*y[i];
  // }
  // *result = local_result[0];  
  // return 0;
}
