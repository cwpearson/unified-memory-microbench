#include <string>
#include <vector>

#include "check_cuda.cuh"

template <typename T>
__global__ void triad_kernel(T *a, const T *b, const T *c, const T scalar,
                             const size_t n) {
  for (size_t i = blockDim.x * blockIdx.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    a[i] = b[i] + scalar * c[i];
  }
}

typedef struct {
  std::vector<double> times;
  std::vector<double> metrics;
  std::string unit;
} Results;

template <typename T>
Results triad(size_t bytes, const T scalar, const size_t numIters) {

  Results results;
  results.unit = "b/s";

  // number of elements and actual allocation size
  const size_t n = bytes / sizeof(T);
  bytes = bytes / sizeof(T) * sizeof(T);

  int *a = nullptr;
  int *b = nullptr;
  int *c = nullptr;

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

int main(void) {

  size_t bytes = 2ul * 1024ul * 1024ul * 1024ul;

  Results results = triad(bytes, 1, 3);

  // print header
  printf("benchmark\tsize\tunit");
  for (size_t i = 0; i < results.times.size(); ++i) {
    printf("\t%lu", i);
  }
  printf("\n");

  // print times
  printf("triad\t%lu\t%s", bytes, "s");
  for (auto t : results.times) {
    printf("\t%f", t);
  }
  printf("\n");

  // print metrics
  printf("triad\t%lu\t%s", bytes, results.unit.c_str());
  for (auto m : results.metrics) {
    printf("\t%f", m);
  }
  printf("\n");
}