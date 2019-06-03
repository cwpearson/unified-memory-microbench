#include <cstdio>
#include <memory>

#include <cxxopts.hpp>

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

int main(int argc, char **argv) {

  cxxopts::Options options("triad", "triad benchmarks");

  options.add_options()
  ("n,num-iters",
   "Number of iterations",
    cxxopts::value<size_t>()->default_value("3"))
  ("b,bytes",
   "Number of bytes",
    cxxopts::value<size_t>()->default_value("2147483648")) // 2G
  ("h,help",
   "Show help")
  ;
  auto result = options.parse(argc, argv);

  bool help = result["help"].as<bool>();

  if (help) {
    printf("%s\n", options.help().c_str());
    exit(0);
  }

  size_t numIters = result["num-iters"].as<size_t>();
  size_t bytes = result["bytes"].as<size_t>();

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