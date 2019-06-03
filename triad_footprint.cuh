#pragma once

#include <functional>

#include "cuda_malloc_managed.cuh"

template <typename T>
__global__ void triad_footprint_kernel(T *a, const T *b, const T *c, const T scalar,
                             const size_t n) {
  for (size_t i = blockDim.x * blockIdx.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    a[i] = b[i] + scalar * c[i];
  }
}

template <typename T, typename Alloc = CUDAMallocManaged<T>>
Results triad_footprint(size_t bytes, const T scalar, const size_t numIters, Alloc alloc = Alloc()) {

  Results results;
  results.unit = "b/s";

  // number of elements and actual allocation size
  const size_t n = bytes / sizeof(T);
  bytes = bytes / sizeof(T) * sizeof(T);

  T *a = alloc.allocate(n);
  T *b = alloc.allocate(n);
  T *c = alloc.allocate(n);

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cudaStream_t stream = nullptr;

  CUDA_RUNTIME(cudaStreamCreate(&stream));
  CUDA_RUNTIME(cudaEventCreate(&start));
  CUDA_RUNTIME(cudaEventCreate(&stop));

  for (size_t iter = 0; iter < numIters; ++iter) {
    // scale each kernel so that it only touches footprint memory
    const size_t footprintBytes = 4ul * 1024ul * 1024ul * 1024ul;
    const size_t footprintElems = footprintBytes / 3 /* number of arrays */ / sizeof(T);

    CUDA_RUNTIME(cudaEventRecord(start, stream));
    for (size_t startIdx = 0; startIdx < n; startIdx += footprintElems) {
      size_t stopIdx = min(startIdx + footprintElems, n);
      size_t kernelN = stopIdx - startIdx;
      T *aBegin = &a[startIdx];
      T *bBegin = &b[startIdx];
      T *cBegin = &c[startIdx];
      triad_footprint_kernel<<<150, 512, 0, stream>>>(aBegin, bBegin, cBegin, scalar, kernelN);
    }
    
    CUDA_RUNTIME(cudaEventRecord(stop, stream));

    CUDA_RUNTIME(cudaStreamSynchronize(stream));
    float elapsed;
    CUDA_RUNTIME(cudaEventElapsedTime(&elapsed, start, stop));

    // add time and metric
    results.times.push_back(elapsed / 1000.0);
    results.metrics.push_back(3ul*bytes / (elapsed / 1000.0));
  }

  CUDA_RUNTIME(cudaStreamDestroy(stream));
  CUDA_RUNTIME(cudaEventDestroy(start));
  CUDA_RUNTIME(cudaEventDestroy(stop));

  alloc.deallocate(a, n);
  alloc.deallocate(b, n);
  alloc.deallocate(c, n);
  a = nullptr;
  b = nullptr;
  c = nullptr;
  return results;
}