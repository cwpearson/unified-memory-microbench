#pragma once

#include <functional>

#include "alignment.hpp"
#include "cuda_malloc_managed.cuh"

template <typename T>
__global__ void triad_footprint_kernel(T *a, const T *b, const T *c,
                                       const T scalar, const size_t n) {
  for (size_t i = blockDim.x * blockIdx.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    a[i] = b[i] + scalar * c[i];
  }
}

template <typename T, typename Alloc = CUDAMallocManaged<T>>
Results triad_footprint(size_t bytes, const T scalar, const size_t numIters,
                        Alloc alloc = Alloc()) {

  const size_t dimBlock = 512;
  int maxActiveBlocks;
  CUDA_RUNTIME(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &maxActiveBlocks, triad_footprint_kernel<T>, dimBlock, 0));
  int device;
  cudaDeviceProp props;
  CUDA_RUNTIME(cudaGetDevice(&device));
  CUDA_RUNTIME(cudaGetDeviceProperties(&props, device));
  const size_t dimGrid = maxActiveBlocks * props.multiProcessorCount;
  fprintf(stderr, "kernel dims: <<<%lu, %lu>>>\n", dimGrid, dimBlock);

  Results results;
  results.unit = "b/s";

  // 3 arrays
  bytes /= 3;

  // number of elements and actual allocation size
  bytes = bytes / sizeof(T) * sizeof(T);
  const size_t n = bytes / sizeof(T);

  size_t alignment = 256;

  T *a = alloc.allocate(n + alignment);
  T *b = alloc.allocate(n + alignment);
  T *c = alloc.allocate(n + alignment);
  T *alignedA = align(a, alignment);
  T *alignedB = align(b, alignment);
  T *alignedC = align(c, alignment);

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cudaStream_t stream = nullptr;

  CUDA_RUNTIME(cudaStreamCreate(&stream));
  CUDA_RUNTIME(cudaEventCreate(&start));
  CUDA_RUNTIME(cudaEventCreate(&stop));

  for (size_t iter = 0; iter < numIters; ++iter) {
    // scale each kernel so that it only touches footprint memory
    const size_t footprintBytes = 3ul * 1024ul * 1024ul * 1024ul;
    const size_t footprintElems =
        footprintBytes / 3 /* number of arrays */ / sizeof(T);

    CUDA_RUNTIME(cudaEventRecord(start, stream));
    for (size_t startIdx = 0; startIdx < n; startIdx += footprintElems) {
      size_t stopIdx = min(startIdx + footprintElems, n);
      size_t kernelN = stopIdx - startIdx;
      T *aBegin = &alignedA[startIdx];
      T *bBegin = &alignedB[startIdx];
      T *cBegin = &alignedC[startIdx];
      fprintf(stderr, "iter %lu: launching: idx [%lu %lu) %lu B each\n", iter, startIdx, stopIdx, kernelN * sizeof(T));
      fprintf(stderr, "align of a %lu b %lu c %lu\n", alignment_of(aBegin), alignment_of(bBegin), alignment_of(cBegin));
      triad_footprint_kernel<<<dimGrid, dimBlock, 0, stream>>>(
          aBegin, bBegin, cBegin, scalar, kernelN);
    }

    CUDA_RUNTIME(cudaEventRecord(stop, stream));

    CUDA_RUNTIME(cudaStreamSynchronize(stream));
    float elapsed;
    CUDA_RUNTIME(cudaEventElapsedTime(&elapsed, start, stop));
    elapsed /= 1000;

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
