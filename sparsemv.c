#include <stdlib.h>
#include <ctype.h>
#include <assert.h>
#include <math.h>

#include <immintrin.h>
#include <omp.h>

#include "sparsemv.h"

/**
 * @brief Compute matrix vector product (y = A*x)
 * 
 * @param A Known matrix
 * @param x Known vector
 * @param y Return vector
 * @return int 0 if no error
 */
int sparsemv(struct mesh *A, const double * const x, double * const y)
{

  // TODO: Optimize this file
  // ! Cpu is not hyperthreaded, hence 'export OMP_NUM_THREADS=6' is used since cpu has 6 logical cores
  // ! There aren't any race conditions so far since waxpby changed with thread number yet is not even multithreaded so Dont worry about that

  const int nrow = (const int) A->local_nrow;
  int j = 0;
  double sum = 0.0;
  #pragma omp parallel for private(j, sum)
  for (int i=0; i< nrow; i++) {
    sum = 0.0;
    const double * const cur_vals = (const double * const) A->ptr_to_vals_in_row[i];
    const int * const cur_inds = (const int * const) A->ptr_to_inds_in_row[i];
    const int cur_nnz = (const int) A->nnz_in_row[i];
    int cur_nnzN = (cur_nnz/3)*3;
    for (j=0; j< cur_nnzN; j+=3) {
      sum += cur_vals[j] * x[cur_inds[j]];
      sum += cur_vals[j+1] * x[cur_inds[j+1]];
      sum += cur_vals[j+2] * x[cur_inds[j+2]];
    }
    for (; j < cur_nnz; j++) {
      sum += cur_vals[j] * x[cur_inds[j]];
    }
    y[i] = sum;
  }
  return 0;

  // Loop fission? Loop Pipelining? Improve locality?
}
