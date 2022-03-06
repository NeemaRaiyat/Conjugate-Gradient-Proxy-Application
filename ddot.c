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
  
  double local_result = 0.0;
  int unroll = (n/4)*4;
  #pragma omp parallel for reduction(+: local_result)
  for (int i=0; i<unroll; i+=4) {
    local_result += x[i]*y[i];
    local_result += x[i+1]*y[i+1];
    local_result += x[i+2]*y[i+2];
    local_result += x[i+3]*y[i+3];
  }
  *result = local_result;
  return 0;
}
