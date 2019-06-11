#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "check_cuda.cuh"

struct Results {
  double kernel;
  double copy;
  double total;
};

template <typename T>
__global__ void triad_kernel(T *__restrict__ a, const T *__restrict__ b,
                             const T *__restrict__ c, const T scalar,
                             const size_t n) {
  for (size_t i = blockDim.x * blockIdx.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    a[i] = b[i] + scalar * c[i];
  }
}

typedef enum {
  PAGEABLE,
  PINNED,
  ZERO_COPY,
  MANAGED,
} AllocationType;

typedef enum {
  NONE = 0x0,
  ACCESS = 0x1,
  PREFETCH = 0x2,
} Hint;

template <typename T>
Results benchmark_naive(size_t n, AllocationType at, Hint hint) {

  T *a_h = nullptr;
  T *b_h = nullptr;
  T *c_h = nullptr;

  if (at == PAGEABLE) {
    a_h = new T[n];
    b_h = new T[n];
    c_h = new T[n];
  } else if (at == PINNED) {
    CUDA_RUNTIME(cudaHostAlloc(&a_h, n * sizeof(T), 0));
    CUDA_RUNTIME(cudaHostAlloc(&b_h, n * sizeof(T), 0));
    CUDA_RUNTIME(cudaHostAlloc(&c_h, n * sizeof(T), 0));
  } else if (at == ZERO_COPY) {
    CUDA_RUNTIME(cudaHostAlloc(&a_h, n * sizeof(T), cudaHostAllocMapped));
    CUDA_RUNTIME(cudaHostAlloc(&b_h, n * sizeof(T), cudaHostAllocMapped));
    CUDA_RUNTIME(cudaHostAlloc(&c_h, n * sizeof(T), cudaHostAllocMapped));
  } else if (at == MANAGED) {
    CUDA_RUNTIME(cudaMallocManaged(&a_h, n * sizeof(T)));
    CUDA_RUNTIME(cudaMallocManaged(&b_h, n * sizeof(T)));
    CUDA_RUNTIME(cudaMallocManaged(&c_h, n * sizeof(T)));
  }

  // touch all pages
  // fprintf(stderr, "touch all pages\n");
  for (size_t i = 0; i < n; i += 32) {
    a_h[i] = i;
    b_h[i] = i;
  }

  // fprintf(stderr, "init dev pointers\n");
  T *a_d = nullptr;
  T *b_d = nullptr;
  T *c_d = nullptr;

  if (at == PAGEABLE || at == PINNED) {
    CUDA_RUNTIME(cudaMalloc(&a_d, sizeof(T) * n));
    CUDA_RUNTIME(cudaMalloc(&b_d, sizeof(T) * n));
    CUDA_RUNTIME(cudaMalloc(&c_d, sizeof(T) * n));
  } else if (at == ZERO_COPY) {
    CUDA_RUNTIME(cudaHostGetDevicePointer(&a_d, a_h, 0));
    CUDA_RUNTIME(cudaHostGetDevicePointer(&b_d, c_h, 0));
    CUDA_RUNTIME(cudaHostGetDevicePointer(&c_d, c_h, 0));
  } else if (at == MANAGED) {
    a_d = a_h;
    b_d = b_h;
    c_d = c_h;
  }

  // fprintf(stderr, "create events\n");
  cudaEvent_t kernelStart, kernelStop;
  cudaEvent_t txStart, txStop;
  cudaEvent_t rxStart, rxStop;
  CUDA_RUNTIME(cudaEventCreate(&kernelStart));
  CUDA_RUNTIME(cudaEventCreate(&kernelStop));
  CUDA_RUNTIME(cudaEventCreate(&txStart));
  CUDA_RUNTIME(cudaEventCreate(&txStop));
  CUDA_RUNTIME(cudaEventCreate(&rxStart));
  CUDA_RUNTIME(cudaEventCreate(&rxStop));

  // fprintf(stderr, "h2d\n");
  CUDA_RUNTIME(cudaEventRecord(txStart));
  if (at == PAGEABLE || at == PINNED) {
    CUDA_RUNTIME(cudaMemcpyAsync(a_d, a_h, sizeof(T) * n, cudaMemcpyDefault));
    CUDA_RUNTIME(cudaMemcpyAsync(b_d, b_h, sizeof(T) * n, cudaMemcpyDefault));
    CUDA_RUNTIME(cudaMemcpyAsync(c_d, c_h, sizeof(T) * n, cudaMemcpyDefault));
  }
  if ((at == MANAGED) && (PREFETCH & hint)) {
    CUDA_RUNTIME(cudaMemPrefetchAsync(a_d, sizeof(T) * n, 0));
    CUDA_RUNTIME(cudaMemPrefetchAsync(b_d, sizeof(T) * n, 0));
    CUDA_RUNTIME(cudaMemPrefetchAsync(c_d, sizeof(T) * n, 0));
  }
  if ((at == MANAGED) && (ACCESS & hint)) {
    CUDA_RUNTIME(
        cudaMemAdvise(a_d, sizeof(T) * n, cudaMemAdviseSetAccessedBy, 0));
    CUDA_RUNTIME(
        cudaMemAdvise(b_d, sizeof(T) * n, cudaMemAdviseSetAccessedBy, 0));
    CUDA_RUNTIME(
        cudaMemAdvise(c_d, sizeof(T) * n, cudaMemAdviseSetAccessedBy, 0));
  }
  CUDA_RUNTIME(cudaEventRecord(txStop));

  int dimBlock = 512;
  int dimGrid = (n + dimBlock - 1) / dimBlock;

  // fprintf(stderr, "launch\n");
  CUDA_RUNTIME(cudaEventRecord(kernelStart));
  triad_kernel<<<dimGrid, dimBlock>>>(a_d, b_d, c_d, 1, n);
  CUDA_RUNTIME(cudaEventRecord(kernelStop));

  // fprintf(stderr, "d2h\n");
  CUDA_RUNTIME(cudaEventRecord(rxStart));
  if (at == PAGEABLE || at == PINNED) {
    CUDA_RUNTIME(cudaMemcpyAsync(c_h, c_d, sizeof(T) * n, cudaMemcpyDefault));
  }
  CUDA_RUNTIME(cudaEventRecord(rxStop));

  // fprintf(stderr, "times\n");
  CUDA_RUNTIME(cudaDeviceSynchronize());
  float txMillis;
  CUDA_RUNTIME(cudaEventElapsedTime(&txMillis, txStart, txStop));
  float rxMillis;
  CUDA_RUNTIME(cudaEventElapsedTime(&rxMillis, rxStart, rxStop));
  float kernelMillis;
  CUDA_RUNTIME(cudaEventElapsedTime(&kernelMillis, kernelStart, kernelStop));
  float totalMillis;
  CUDA_RUNTIME(cudaEventElapsedTime(&totalMillis, txStart, rxStop));

  // fprintf(stderr, "cuda free\n");
  if (at == PAGEABLE || at == PINNED) {
    CUDA_RUNTIME(cudaFree(a_d));
    CUDA_RUNTIME(cudaFree(b_d));
    CUDA_RUNTIME(cudaFree(c_d));
  }

  // fprintf(stderr, "host free\n");
  if (at == PAGEABLE) {
    delete[] a_h;
    delete[] b_h;
    delete[] c_h;
  } else if (at == PINNED || at == ZERO_COPY) {
    CUDA_RUNTIME(cudaFreeHost(a_h));
    CUDA_RUNTIME(cudaFreeHost(b_h));
    CUDA_RUNTIME(cudaFreeHost(c_h));
  } else if (at == MANAGED) {
    CUDA_RUNTIME(cudaFree(a_h));
    CUDA_RUNTIME(cudaFree(b_h));
    CUDA_RUNTIME(cudaFree(c_h));
  }

  // fprintf(stderr, "destroy event\n");
  CUDA_RUNTIME(cudaEventDestroy(kernelStart));
  CUDA_RUNTIME(cudaEventDestroy(kernelStop));
  CUDA_RUNTIME(cudaEventDestroy(txStart));
  CUDA_RUNTIME(cudaEventDestroy(txStop));
  CUDA_RUNTIME(cudaEventDestroy(rxStart));
  CUDA_RUNTIME(cudaEventDestroy(rxStop));

  double copyPerf =
      1000.0 * n * sizeof(T) * 3 / (txMillis + rxMillis) / 1024 / 1024;
  double kernelPerf = 1000.0 * n * sizeof(T) * 3 / kernelMillis / 1024 / 1024;
  double totalPerf = 1000.0 * n * sizeof(T) * 3 / totalMillis / 1024 / 1024;

  // no copies in some of these
  if (at == ZERO_COPY ) {
    copyPerf = -1;
  }
  if ((at == MANAGED) && (hint == NONE)) {
    copyPerf = -1;
  }

  Results results;
  results.kernel = kernelPerf;
  results.copy = copyPerf;
  results.total = totalPerf;
  // printf("%f.2 %f.2 %f.2\n", copyPerf, kernelPerf, totalPerf);
  return results;
}

void print_results(const std::vector<Results> runs, const std::string &sep) {

  for (auto &run : runs) {
    printf("%s", sep.c_str());
    if (run.copy >= 0) {
      printf("%.2e", run.copy);
    }
  }
  for (auto &run : runs) {
    printf("%s", sep.c_str());
    if (run.kernel >= 0) {
      printf("%.2e", run.kernel);
    }
  }
  for (auto &run : runs) {
    printf("%s", sep.c_str());
    if (run.total >= 0) {
      printf("%.2e", run.total);
    }
  }
  std::cout << std::endl;
}

template <typename T> std::vector<Results> run(size_t iters, T fn) {
  std::vector<Results> runs;
  for (size_t i = 0; i < iters; ++i) {
    auto results = fn();
    runs.push_back(results);
  }
  return runs;
}

int main(int argc, char **arvg) {

  std::string sep = ",";
  size_t iters = 5;

  // print header
  std::cout << "n" << sep << "bmark";
  for (size_t i = 0; i < iters; ++i) {
    std::cout << sep << "copy_" + std::to_string(i);
  }
  for (size_t i = 0; i < iters; ++i) {
    std::cout << sep << "kernel_" + std::to_string(i);
  }
  for (size_t i = 0; i < iters; ++i) {
    std::cout << sep << "total_" + std::to_string(i);
  }
  std::cout << std::endl;

  // runs
  for (size_t n = 1e5; n <= 1e7; n *= 1.3) {

    std::vector<Results> runs;
    runs = run(iters, std::bind(benchmark_naive<int>, n, PAGEABLE, NONE));
    printf("%.2e%s%s", (double)n, sep.c_str(), "pageable          ");
    print_results(runs, sep);

    runs = run(iters, std::bind(benchmark_naive<int>, n, PINNED, NONE));
    printf("%.2e%s%s", (double)n, sep.c_str(), "pinned            ");
    print_results(runs, sep);

    runs = run(iters, std::bind(benchmark_naive<int>, n, ZERO_COPY, NONE));
    printf("%.2e%s%s", (double)n, sep.c_str(), "zero-copy         ");
    print_results(runs, sep);

    runs = run(iters, std::bind(benchmark_naive<int>, n, MANAGED, NONE));
    printf("%.2e%s%s", (double)n, sep.c_str(), "um                ");
    print_results(runs, sep);

    runs = run(iters, std::bind(benchmark_naive<int>, n, MANAGED, ACCESS));
    printf("%.2e%s%s", (double)n, sep.c_str(), "um-access         ");
    print_results(runs, sep);

    runs = run(iters, std::bind(benchmark_naive<int>, n, MANAGED, PREFETCH));
    printf("%.2e%s%s", (double)n, sep.c_str(), "um-prefetch       ");
    print_results(runs, sep);

    runs = run(iters, std::bind(benchmark_naive<int>, n, MANAGED,
                                Hint(ACCESS | PREFETCH)));
    printf("%.2e%s%s", (double)n, sep.c_str(), "um-access-prefetch");
    print_results(runs, sep);
  }
}