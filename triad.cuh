#pragma once

#include <functional>

#include "alignment.hpp"
#include "cuda_malloc_managed.cuh"

template <typename T>
__global__ void triad_kernel(T *a, const T *b, const T *c, const T scalar,
                             const size_t n) {
  for (size_t i = blockDim.x * blockIdx.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    a[i] = b[i] + scalar * c[i];
  }
}

template <typename T, typename Alloc = CUDAMallocManaged<T>>
Results triad(size_t bytes, const T scalar, const size_t numIters,
              Alloc alloc = Alloc()) {

  const size_t dimBlock = 512;
  int maxActiveBlocks;
  CUDA_RUNTIME(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &maxActiveBlocks, triad_kernel<T>, dimBlock, 0));
  int device;
  cudaDeviceProp props;
  CUDA_RUNTIME(cudaGetDevice(&device));
  CUDA_RUNTIME(cudaGetDeviceProperties(&props, device));
  const size_t dimGrid = maxActiveBlocks * props.multiProcessorCount;
  fprintf(stderr, "kernel dims: <<<%lu, %lu>>>\n", dimGrid, dimBlock);

  Results results;
  results.unit = "b/s";

  // we do 3 arrays, so each one should be bytes/3 for a total footprint of
  // bytes
  bytes /= 3;

  // number of elements and actual allocation size
  bytes = bytes / sizeof(T) * sizeof(T);
  const size_t n = bytes / sizeof(T);

  // ensure arrays are aligned to this address
  size_t alignment = 256;

  T *a = alloc.allocate(n + alignment);
  T *b = alloc.allocate(n + alignment);
  T *c = alloc.allocate(n + alignment);

  T *alignedA = align(a, alignment);
  T *alignedB = align(b, alignment);
  T *alignedC = align(c, alignment);

  printf("alignment a %lu b %lu c %lu\n", alignment_of(alignedA), alignment_of(alignedB), alignment_of(alignedC));

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cudaStream_t stream = nullptr;

  CUDA_RUNTIME(cudaStreamCreate(&stream));
  CUDA_RUNTIME(cudaEventCreate(&start));
  CUDA_RUNTIME(cudaEventCreate(&stop));

  for (size_t iter = 0; iter < numIters; ++iter) {
    fprintf(stderr, "launching [%lu %lu)\n", 0ul, n);
    CUDA_RUNTIME(cudaEventRecord(start, stream));
    triad_kernel<<<dimGrid, dimBlock, 0, stream>>>(alignedA, alignedB, alignedC, scalar, n);
    CUDA_RUNTIME(cudaEventRecord(stop, stream));

    CUDA_RUNTIME(cudaStreamSynchronize(stream));
    float elapsed;
    CUDA_RUNTIME(cudaEventElapsedTime(&elapsed, start, stop));
    elapsed /= 1000.0; // convert to seconds

    // add time and metric
    results.times.push_back(elapsed);
    results.metrics.push_back(3ul * bytes / elapsed);
  }

  CUDA_RUNTIME(cudaStreamDestroy(stream));
  CUDA_RUNTIME(cudaEventDestroy(start));
  CUDA_RUNTIME(cudaEventDestroy(stop));

  alloc.deallocate(a, n + alignment);
  alloc.deallocate(b, n + alignment);
  alloc.deallocate(c, n + alignment);
  a = nullptr;
  b = nullptr;
  c = nullptr;
  return results;
}
