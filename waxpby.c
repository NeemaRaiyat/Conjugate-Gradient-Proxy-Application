#include "waxpby.h"
// #include <GL/glut.h>
#include <immintrin.h>
#include <omp.h>
#include <stdio.h>

#include <pthread.h>

#define NUMTHREADS 6

struct thread_arg {
  int lower;
  int upper;
  double * x;
  double beta;
  double * y;
  double * w;
};

void * thread_function(void * arg) {
  struct thread_arg * t_arg = (struct thread_arg *) arg;
  // x, y and w are aligned, look in generate_matrix.c, so we dont need to use things like storeu and loadu
  // printf("t0");

  int i = 0;
  int loopN = (t_arg->upper/4)*4;
  __m256d betaVec = _mm256_set1_pd(t_arg->beta);
  // printf("t1");

  for (i = t_arg->lower; i<loopN; i+=4) {
    __m256d xVec = _mm256_load_pd(t_arg->x + i);
    __m256d yVec = _mm256_load_pd(t_arg->y + i);
    _mm256_store_pd(t_arg->w + i, _mm256_add_pd(xVec, _mm256_mul_pd(betaVec, yVec)));
  }
  for (; i<t_arg->upper; i++) {
    t_arg->w[i] = t_arg->x[i] + t_arg->beta * t_arg->y[i];
  }
  return NULL;
}

/**
 * @brief Compute the update of a vector with the sum of two scaled vectors
 * 
 * @param n Number of vector elements
 * @param x Input vector
 * @param beta Scalars applied to y
 * @param y Input vector
 * @param w Output vector
 * @return int 0 if no error
 */
int waxpby (const int n, double * const x, const double beta, double * const y, double * const w) {  
  
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

  // return 0;

  // printf("a0");

  static pthread_t tid[NUMTHREADS];
  int thread_range = n/6;
  int leftover_range = n - ((n/6)*6);
  struct thread_arg t_arg;
  t_arg.beta = beta;
  // printf("a1");
  for (int i = 0; i < NUMTHREADS; i++) {
  // printf("a2");
    
    t_arg.lower = i * thread_range;
    t_arg.upper = t_arg.lower + thread_range;
    if (i == NUMTHREADS - 1) {
      t_arg.upper = t_arg.lower + leftover_range;
    }
    t_arg.x = x;
    t_arg.y = y;
    t_arg.w = w;
  // printf("a3");

    if (pthread_create(&tid[i], NULL, thread_function, (void *) &t_arg)) {
        fprintf(stderr, "%s","\nERROR: Could not create thread\n");
        exit(EXIT_FAILURE);
    }
  }

  return 0;
}

