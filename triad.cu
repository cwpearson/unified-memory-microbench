#include <cstdio>

#include "check_cuda.cuh"
#include "results.hpp"



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
    results.times.push_back(elapsed);
    results.metrics.push_back(3ul*bytes / elapsed);
  }

  CUDA_RUNTIME(cudaStreamDestroy(stream));
  CUDA_RUNTIME(cudaEventDestroy(start));
  CUDA_RUNTIME(cudaEventDestroy(stop));

  CUDA_RUNTIME(cudaFree(a));
  a = nullptr;
  CUDA_RUNTIME(cudaFree(b));
  b = nullptr;
  CUDA_RUNTIME(cudaFree(c));
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
    results.times.push_back(elapsed);
    results.metrics.push_back(3ul*bytes / elapsed);
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
    CUDA_RUNTIME(cudaEventRecord(start, stream));

    // scale each kernel so that it only touches footprint memory
    const size_t footprint = 4ul * 1024ul * 1024ul * 1024ul;

    for (size_t startIdx = 0; startIdx < (footprint + sizeof(T) - 1) / sizeof(T); startIdx += footprint) {
      size_t stopIdx = min(startIdx + footprint, n);
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
    results.times.push_back(elapsed);
    results.metrics.push_back(3ul*bytes / elapsed);
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

void print(const std::string &name, const size_t bytes, const Results &results) {
  // print header
  printf("benchmark\tsize\tunit");
  for (size_t i = 0; i < results.times.size(); ++i) {
    printf("\t%lu", i);
  }
  printf("\n");

  // print times
  printf("%s\t%lu\t%s", name.c_str(), bytes, "s");
  for (auto t : results.times) {
    printf("\t%f", t);
  }
  printf("\n");

  // print metrics
  printf("%s\t%lu\t%s", name.c_str(), bytes, results.unit.c_str());
  for (auto m : results.metrics) {
    printf("\t%f", m);
  }
  printf("\n");
}

int main(void) {

  size_t bytes = 2ul * 1024ul * 1024ul * 1024ul;

  Results results = triad<int>(bytes, 1, 3);
  print("triad", bytes, results);

  results = triad_system<int>(bytes, 1, 3);
  print("triad_system", bytes, results);

  results = triad_footprint_system<int>(bytes, 1, 3);
  print("triad_footprint_system", bytes, results);

}