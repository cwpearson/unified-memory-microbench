#include <cstdio>
#include <memory>

#include "check_cuda.cuh"
#include "results.hpp"
#include "cuda_malloc_managed.cuh"
#include "triad_footprint.cuh"
#include "triad.cuh"


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

  results = triad<int, CUDAMallocManaged<int>>(bytes, 1, numIters);
  print("triad/cudamallocmanaged", bytes, results);

  results = triad_footprint<int, CUDAMallocManaged<int>>(bytes, 1, numIters);
  print("triad_footprint/cudamallocmanaged", bytes, results);

  results = triad<int, std::allocator<int>>(bytes, 1, numIters);
  print("triad/system", bytes, results);

  results = triad_footprint<int, std::allocator<int>>(bytes, 1, numIters);
  print("triad_footprint/system", bytes, results);

}