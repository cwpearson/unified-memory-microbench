#pragma once

#include <functional>

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
Results triad(size_t bytes, const T scalar, const size_t numIters, Alloc alloc = Alloc()) {

  Results results;
  results.unit = "b/s";

  // we do 3 arrays, so each one should be bytes/3 for a total footprint of bytes
  bytes /= 3;

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
    CUDA_RUNTIME(cudaEventRecord(start, stream));
    triad_kernel<<<150, 512, 0, stream>>>(a, b, c, scalar, n);
    CUDA_RUNTIME(cudaEventRecord(stop, stream));

    CUDA_RUNTIME(cudaStreamSynchronize(stream));
    float elapsed;
    CUDA_RUNTIME(cudaEventElapsedTime(&elapsed, start, stop));
    elapsed /= 1000.0; // convert to seconds

    // add time and metric
    results.times.push_back(elapsed);
    results.metrics.push_back(3ul*bytes / elapsed);
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

