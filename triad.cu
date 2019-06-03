#include <cstdio>

#include "check_cuda.cuh"
#include "results.hpp"
#include "allocator.cuh"



template <typename T>
__global__ void triad_kernel(T *a, const T *b, const T *c, const T scalar,
                             const size_t n) {
  for (size_t i = blockDim.x * blockIdx.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    a[i] = b[i] + scalar * c[i];
  }
}

template <typename T>
Results triad(size_t bytes, const T scalar, const size_t numIters) {

  Results results;
  results.unit = "b/s";

  // number of elements and actual allocation size
  const size_t n = bytes / sizeof(T);
  bytes = bytes / sizeof(T) * sizeof(T);

  T *a = nullptr;
  T *b = nullptr;
  T *c = nullptr;

  CUDA_RUNTIME(cudaMallocManaged(&a, bytes));
  CUDA_RUNTIME(cudaMallocManaged(&b, bytes));
  CUDA_RUNTIME(cudaMallocManaged(&c, bytes));

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

    // add time and metric
    results.times.push_back(elapsed / 1000.0);
    results.metrics.push_back(3ul*bytes / (elapsed / 1000.0));
  }

  CUDA_RUNTIME(cudaStreamDestroy(stream));
  CUDA_RUNTIME(cudaEventDestroy(start));
  CUDA_RUNTIME(cudaEventDestroy(stop));

  CUDA_RUNTIME(cudaFree(a));
  CUDA_RUNTIME(cudaFree(b));
  CUDA_RUNTIME(cudaFree(c));
  a = nullptr;
  b = nullptr;
  c = nullptr;
  return results;
}

template <typename T>
Results triad_system(size_t bytes, const T scalar, const size_t numIters) {

  Results results;
  results.unit = "b/s";

  // number of elements and actual allocation size
  const size_t n = bytes / sizeof(T);
  bytes = bytes / sizeof(T) * sizeof(T);

  T *a = nullptr;
  T *b = nullptr;
  T *c = nullptr;

  a = (T*)malloc(bytes);
  b = (T*)malloc(bytes);
  c = (T*)malloc(bytes);

  if (a == nullptr) {
    return results;
  }
  if (b == nullptr) {
    return results;
  }
  if (c == nullptr) {
    return results;
  }


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

    // add time and metric
    results.times.push_back(elapsed / 1000.0);
    results.metrics.push_back(3ul*bytes / (elapsed / 1000.0));
  }

  CUDA_RUNTIME(cudaStreamDestroy(stream));
  CUDA_RUNTIME(cudaEventDestroy(start));
  CUDA_RUNTIME(cudaEventDestroy(stop));

  free(a);
  a = nullptr;
  free(b);
  b = nullptr;
  free(c);
  c = nullptr;
  return results;
}

template <typename T>
Results triad_footprint(size_t bytes, const T scalar, const size_t numIters) {

  Results results;
  results.unit = "b/s";

  // number of elements and actual allocation size
  const size_t n = bytes / sizeof(T);
  bytes = bytes / sizeof(T) * sizeof(T);

  T *a = nullptr;
  T *b = nullptr;
  T *c = nullptr;
  CUDA_RUNTIME(cudaMallocManaged(&a, bytes));
  CUDA_RUNTIME(cudaMallocManaged(&b, bytes));
  CUDA_RUNTIME(cudaMallocManaged(&c, bytes));

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
      triad_kernel<<<150, 512, 0, stream>>>(aBegin, bBegin, cBegin, scalar, kernelN);
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

  CUDA_RUNTIME(cudaFree(a));
  CUDA_RUNTIME(cudaFree(b));
  CUDA_RUNTIME(cudaFree(c));
  a = nullptr;
  b = nullptr;
  c = nullptr;
  return results;
}

template <typename T>
Results triad_footprint_system(size_t bytes, const T scalar, const size_t numIters) {

  Results results;
  results.unit = "b/s";

  // number of elements and actual allocation size
  const size_t n = bytes / sizeof(T);
  bytes = bytes / sizeof(T) * sizeof(T);

  T *a = nullptr;
  T *b = nullptr;
  T *c = nullptr;

  a = (T*)malloc(bytes);
  b = (T*)malloc(bytes);
  c = (T*)malloc(bytes);

  if (a == nullptr) {
    return results;
  }
  if (b == nullptr) {
    return results;
  }
  if (c == nullptr) {
    return results;
  }


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
      triad_kernel<<<150, 512, 0, stream>>>(aBegin, bBegin, cBegin, scalar, kernelN);
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

  free(a);
  a = nullptr;
  free(b);
  b = nullptr;
  free(c);
  c = nullptr;
  return results;
}

void print_header(size_t numIters) {
  printf("benchmark\tsize\tunit");
  for (size_t i = 0; i < numIters; ++i) {
    printf("\t%lu", i);
  }
  printf("\n");
}

void print(const std::string &name, const size_t bytes, const Results &results) {
  // print times
  printf("%s\t%lu\t%s", name.c_str(), bytes, "s");
  for (auto t : results.times) {
    printf("\t%.1e", t);
  }
  printf("\n");

  // print metrics
  printf("%s\t%lu\t%s", name.c_str(), bytes, results.unit.c_str());
  for (auto m : results.metrics) {
    printf("\t%.1e", m);
  }
  printf("\n");
}

int main(void) {

  size_t bytes = 2ul * 1024ul * 1024ul * 1024ul;
  size_t numIters = 3;

  print_header(numIters);

  Results results;

  results = triad<int>(bytes, 1, numIters);
  print("triad", bytes, results);

  results = triad_footprint<int>(bytes, 1, numIters);
  print("triad_footprint", bytes, results);

  results = triad_system<int>(bytes, 1, numIters);
  print("triad_system", bytes, results);

  results = triad_footprint_system<int>(bytes, 1, numIters);
  print("triad_footprint_system", bytes, results);

}