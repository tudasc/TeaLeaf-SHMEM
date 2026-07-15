#include "comms.h"
#include "settings.h"

#include <shmem.h>

#define REDUCE_WRK_SIZE 8

static double *reduce_src_dbl = nullptr;
static double *reduce_dst_dbl = nullptr;
static long *reduce_src_long = nullptr;
static long *reduce_dst_long = nullptr;
static int *reduce_src_int = nullptr;
static int *reduce_dst_int = nullptr;

static double *pWrk_dbl = nullptr;
static long *pWrk_long = nullptr;
static int *pWrk_int = nullptr;
static long *pSync = nullptr;

// Initialise OpenSHMEM
void initialise_comms(int, char **) {
  shmem_init();

  reduce_src_dbl = static_cast<double *>(shmem_malloc(sizeof(double)));
  reduce_dst_dbl = static_cast<double *>(shmem_malloc(sizeof(double)));
  reduce_src_long = static_cast<long *>(shmem_malloc(sizeof(long)));
  reduce_dst_long = static_cast<long *>(shmem_malloc(sizeof(long)));
  reduce_src_int = static_cast<int *>(shmem_malloc(sizeof(int)));
  reduce_dst_int = static_cast<int *>(shmem_malloc(sizeof(int)));

  pWrk_dbl = static_cast<double *>(shmem_malloc(sizeof(double) * REDUCE_WRK_SIZE));
  pWrk_long = static_cast<long *>(shmem_malloc(sizeof(long) * REDUCE_WRK_SIZE));
  pWrk_int = static_cast<int *>(shmem_malloc(sizeof(int) * REDUCE_WRK_SIZE));
  pSync = static_cast<long *>(shmem_malloc(sizeof(long) * SHMEM_REDUCE_SYNC_SIZE));

  for (int i = 0; i < SHMEM_REDUCE_SYNC_SIZE; ++i) {
    pSync[i] = SHMEM_SYNC_VALUE;
  }

  // Ensure pSync is initialised on every PE before the first collective.
  shmem_barrier_all();
}

// Initialise the rank information
void initialise_ranks(Settings &settings) {
  settings.rank = shmem_my_pe();
  settings.num_ranks = shmem_n_pes();
}

// Teardown OpenSHMEM
void finalise_comms() {
  shmem_free(reduce_src_dbl);
  shmem_free(reduce_dst_dbl);
  shmem_free(reduce_src_long);
  shmem_free(reduce_dst_long);
  shmem_free(reduce_src_int);
  shmem_free(reduce_dst_int);
  shmem_free(pWrk_dbl);
  shmem_free(pWrk_long);
  shmem_free(pWrk_int);
  shmem_free(pSync);
  shmem_finalize();
}

// One-sided, non-blocking put of a halo buffer into the target PE's symmetric recv buffer.
void put_halo_message(Settings &settings, double *target_recv_buffer, double *send_buffer, int buffer_len, int target_pe) {
  START_PROFILING(settings.kernel_profile);
  shmem_double_put_nbi(target_recv_buffer, send_buffer, buffer_len, target_pe);
  STOP_PROFILING(settings.kernel_profile, __func__);
}

// Barrier that completes all outstanding halo puts before any PE proceeds
void halo_sync(Settings &settings) {
  START_PROFILING(settings.kernel_profile);
  shmem_barrier_all();
  STOP_PROFILING(settings.kernel_profile, __func__);
}

// Reduce over all PEs to get sum
void sum_over_ranks(Settings &settings, double *a) {
  START_PROFILING(settings.kernel_profile);
  *reduce_src_dbl = *a;
  shmem_double_sum_to_all(reduce_dst_dbl, reduce_src_dbl, 1, 0, 0, shmem_n_pes(), pWrk_dbl, pSync);
  *a = *reduce_dst_dbl;
  STOP_PROFILING(settings.kernel_profile, __func__);
}

// Reduce across all PEs to get minimum value
void min_over_ranks(Settings &settings, double *a) {
  START_PROFILING(settings.kernel_profile);
  *reduce_src_dbl = *a;
  shmem_double_min_to_all(reduce_dst_dbl, reduce_src_dbl, 1, 0, 0, shmem_n_pes(), pWrk_dbl, pSync);
  *a = *reduce_dst_dbl;
  STOP_PROFILING(settings.kernel_profile, __func__);
}

// Integer sum reduction (used for reporting aggregate buffer sizes)
void sum_over_ranks(Settings &settings, long *a) {
  START_PROFILING(settings.kernel_profile);
  *reduce_src_long = *a;
  shmem_long_sum_to_all(reduce_dst_long, reduce_src_long, 1, 0, 0, shmem_n_pes(), pWrk_long, pSync);
  *a = *reduce_dst_long;
  STOP_PROFILING(settings.kernel_profile, __func__);
}

// Integer max reduction (used to size symmetric halo buffers uniformly)
void max_over_ranks(Settings &settings, int *a) {
  START_PROFILING(settings.kernel_profile);
  *reduce_src_int = *a;
  shmem_int_max_to_all(reduce_dst_int, reduce_src_int, 1, 0, 0, shmem_n_pes(), pWrk_int, pSync);
  *a = *reduce_dst_int;
  STOP_PROFILING(settings.kernel_profile, __func__);
}

// Synchronise all PEs
void barrier() { shmem_barrier_all(); }

// End the application
void abort_comms() { shmem_global_exit(1); }
