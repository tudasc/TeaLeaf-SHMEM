#pragma once

#include "chunk.h"
#include "settings.h"

void barrier();
void abort_comms();
void finalise_comms();
void initialise_comms(int argc, char **argv);
void initialise_ranks(Settings &settings);

// Global reductions
void sum_over_ranks(Settings &settings, double *a);
void min_over_ranks(Settings &settings, double *a);
void sum_over_ranks(Settings &settings, long *a);
void max_over_ranks(Settings &settings, int *a);

void put_halo_message(Settings &settings, double *target_recv_buffer, double *send_buffer, int buffer_len, int target_pe);

void halo_sync(Settings &settings);
