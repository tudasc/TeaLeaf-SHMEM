#include "chunk.h"
#include "comms.h"
#include "drivers.h"
#include "kernel_interface.h"

// Attempts to pack buffers
int invoke_pack_or_unpack(Chunk *chunk, Settings &settings, int face, int depth, int offset, bool pack, FieldBufferType buffer) {
  int buffer_len = 0;

  for (int ii = 0; ii < NUM_FIELDS; ++ii) {
    if (!settings.fields_to_exchange[ii]) {
      continue;
    }

    FieldBufferType field;
    switch (ii) {
      case FIELD_DENSITY: field = chunk->density; break;
      case FIELD_ENERGY0: field = chunk->energy0; break;
      case FIELD_ENERGY1: field = chunk->energy; break;
      case FIELD_U: field = chunk->u; break;
      case FIELD_P: field = chunk->p; break;
      case FIELD_SD: field = chunk->sd; break;
      default: die(__LINE__, __FILE__, "Incorrect field provided: %d.\n", ii + 1);
    }

    if (settings.kernel_language == Kernel_Language::C) {
      run_pack_or_unpack(chunk, settings, depth, face, pack, field, buffer, buffer_len);
    } else if (settings.kernel_language == Kernel_Language::FORTRAN) {
    }
    buffer_len += depth * offset;
  }

  return buffer_len;
}

// Maps a global chunk index to the PE that owns it.
static inline int chunk_owner_pe(Settings &settings, int global_chunk) { return global_chunk / settings.num_chunks_per_rank; }
static inline int chunk_local_index(Settings &settings, int global_chunk) { return global_chunk % settings.num_chunks_per_rank; }

// Invokes the kernels that perform remote halo exchanges
void remote_halo_driver(Chunk *chunks, Settings &settings, int depth) {
  for (int cc = 0; cc < settings.num_chunks_per_rank; ++cc) {
    if (chunks[cc].neighbours[CHUNK_LEFT] != EXTERNAL_FACE)
      invoke_pack_or_unpack(&(chunks[cc]), settings, CHUNK_LEFT, depth, chunks[cc].y, true, chunks[cc].left_send);
    if (chunks[cc].neighbours[CHUNK_RIGHT] != EXTERNAL_FACE)
      invoke_pack_or_unpack(&(chunks[cc]), settings, CHUNK_RIGHT, depth, chunks[cc].y, true, chunks[cc].right_send);
  }

  for (int cc = 0; cc < settings.num_chunks_per_rank; ++cc) {
    int buffer_len = 0;
    for (int ii = 0; ii < NUM_FIELDS; ++ii) {
      if (!settings.fields_to_exchange[ii]) continue;
      buffer_len += depth * chunks[cc].y;
    }
    if (chunks[cc].neighbours[CHUNK_LEFT] != EXTERNAL_FACE) {
      int nb = chunks[cc].neighbours[CHUNK_LEFT];
      int lc = chunk_local_index(settings, nb);
      put_halo_message(settings, chunks[lc].right_recv, chunks[cc].left_send, buffer_len, chunk_owner_pe(settings, nb));
    }
    if (chunks[cc].neighbours[CHUNK_RIGHT] != EXTERNAL_FACE) {
      int nb = chunks[cc].neighbours[CHUNK_RIGHT];
      int lc = chunk_local_index(settings, nb);
      put_halo_message(settings, chunks[lc].left_recv, chunks[cc].right_send, buffer_len, chunk_owner_pe(settings, nb));
    }
  }

  halo_sync(settings);

  for (int cc = 0; cc < settings.num_chunks_per_rank; ++cc) {
    if (chunks[cc].neighbours[CHUNK_LEFT] != EXTERNAL_FACE)
      invoke_pack_or_unpack(&(chunks[cc]), settings, CHUNK_LEFT, depth, chunks[cc].y, false, chunks[cc].left_recv);
    if (chunks[cc].neighbours[CHUNK_RIGHT] != EXTERNAL_FACE)
      invoke_pack_or_unpack(&(chunks[cc]), settings, CHUNK_RIGHT, depth, chunks[cc].y, false, chunks[cc].right_recv);
  }
  for (int cc = 0; cc < settings.num_chunks_per_rank; ++cc) {
    if (chunks[cc].neighbours[CHUNK_BOTTOM] != EXTERNAL_FACE)
      invoke_pack_or_unpack(&(chunks[cc]), settings, CHUNK_BOTTOM, depth, chunks[cc].x, true, chunks[cc].bottom_send);
    if (chunks[cc].neighbours[CHUNK_TOP] != EXTERNAL_FACE)
      invoke_pack_or_unpack(&(chunks[cc]), settings, CHUNK_TOP, depth, chunks[cc].x, true, chunks[cc].top_send);
  }

  for (int cc = 0; cc < settings.num_chunks_per_rank; ++cc) {
    int buffer_len = 0;
    for (int ii = 0; ii < NUM_FIELDS; ++ii) {
      if (!settings.fields_to_exchange[ii]) continue;
      buffer_len += depth * chunks[cc].x;
    }

    if (chunks[cc].neighbours[CHUNK_BOTTOM] != EXTERNAL_FACE) {
      int nb = chunks[cc].neighbours[CHUNK_BOTTOM];
      int lc = chunk_local_index(settings, nb);
      put_halo_message(settings, chunks[lc].top_recv, chunks[cc].bottom_send, buffer_len, chunk_owner_pe(settings, nb));
    }
    if (chunks[cc].neighbours[CHUNK_TOP] != EXTERNAL_FACE) {
      int nb = chunks[cc].neighbours[CHUNK_TOP];
      int lc = chunk_local_index(settings, nb);
      put_halo_message(settings, chunks[lc].bottom_recv, chunks[cc].top_send, buffer_len, chunk_owner_pe(settings, nb));
    }
  }

  halo_sync(settings);

  for (int cc = 0; cc < settings.num_chunks_per_rank; ++cc) {
    if (chunks[cc].neighbours[CHUNK_BOTTOM] != EXTERNAL_FACE)
      invoke_pack_or_unpack(&(chunks[cc]), settings, CHUNK_BOTTOM, depth, chunks[cc].x, false, chunks[cc].bottom_recv);
    if (chunks[cc].neighbours[CHUNK_TOP] != EXTERNAL_FACE)
      invoke_pack_or_unpack(&(chunks[cc]), settings, CHUNK_TOP, depth, chunks[cc].x, false, chunks[cc].top_recv);
  }
}
