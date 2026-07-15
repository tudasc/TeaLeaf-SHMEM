#include "chunk.h"
#include "comms.h"
#include "kernel_interface.h"

// Invokes the kernel initialisation kernels
void kernel_initialise_driver(Chunk *chunks, Settings &settings) {
  int lr_len = 0;
  int tb_len = 0;
  for (int cc = 0; cc < settings.num_chunks_per_rank; ++cc) {
    lr_len = tealeaf_MAX(lr_len, chunks[cc].y * settings.halo_depth * NUM_FIELDS);
    tb_len = tealeaf_MAX(tb_len, chunks[cc].x * settings.halo_depth * NUM_FIELDS);
  }
  max_over_ranks(settings, &lr_len);
  max_over_ranks(settings, &tb_len);

  for (int cc = 0; cc < settings.num_chunks_per_rank; ++cc) {
    if (settings.kernel_language == Kernel_Language::C) {
      run_kernel_initialise(&(chunks[cc]), settings, lr_len, tb_len);
    } else if (settings.kernel_language == Kernel_Language::FORTRAN) {
    }
  }
}

// Invokes the kernel finalisation drivers
void kernel_finalise_driver(Chunk *chunks, Settings &settings) {
  for (int cc = 0; cc < settings.num_chunks_per_rank; ++cc) {
    if (settings.kernel_language == Kernel_Language::C) {
      run_kernel_finalise(&(chunks[cc]), settings);
    } else if (settings.kernel_language == Kernel_Language::FORTRAN) {
    }
  }
}
