#include "ddot.h"
#include <omp.h>

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
  
  // Look at slides 61+
  double local_result = 0.0;
  #pragma omp parallel for reduction(+: local_result)
  for (int i=0; i<n; i++) {
    local_result += x[i]*y[i];
  }
  *result = local_result;
  return 0;
}
