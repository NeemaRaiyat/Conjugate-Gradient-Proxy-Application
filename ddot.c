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
  double local_result = 0.0;
  // if (y==x){
  //   for (int i=0; i<n; i++) {
  //     local_result += x[i]*x[i];
  //   }
  // } else {
  //   for (int i=0; i<n; i++) {
  //     local_result += x[i]*y[i];
  //   }
  // }

  // #pragma omp parallel for shared(local_result) reduction(+: local_result)
  // #pragma omp parallel for shared(local_result, x, y) reduction(+: local_result)
  // #pragma omp parallel reduction(+: local_result)
  #pragma omp parallel for reduction(+: local_result)
  for (int i=0; i<n; i++) {
    local_result += x[i]*y[i];
  }

  // REDUCTIONS???
  // #pragma omp parallel for shared(local_result) reduction(+: local_result)
  // #pragma omp parallel, reduction(+: local_result)

  // int i = 0;
  // int loopN = (n/2)*2;
  // __m128d lrVec = _mm_set1_pd(local_result);
  // for (i = 0; i < loopN; i+=2) {
  //   __m128d xVec = _mm_load_pd(x + i);
  //   __m128d yVec = _mm_load_pd(y + i);
  //   lrVec = _mm_add_pd(lrVec, _mm_mul_pd(xVec, yVec));
  // }
  // for (; i < n; i++) {
  //   local_result += x[i]*y[i];
  // }
  // _mm_store_pd(&local_result, lrVec);

  *result = local_result;

  return 0;
}
