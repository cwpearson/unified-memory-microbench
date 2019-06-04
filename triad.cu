#include <cstdio>
#include <memory>

#include <cxxopts.hpp>

#include "check_cuda.cuh"
#include "results.hpp"
#include "cuda_malloc_managed.cuh"
#include "triad_footprint.cuh"
#include "triad.cuh"


void print_header(size_t numIters, const std::string &sep) {
  printf("benchmark%ssize%sunit", sep.c_str(), sep.c_str());
  for (size_t i = 0; i < numIters; ++i) {
    printf("%s%lu", sep.c_str(), i);
  }
  printf("\n");
}

void print(const std::string &name, const size_t bytes, const Results &results, const std::string &sep) {
  // print times
  //printf("%s\t%lu\t%s", name.c_str(), bytes, "s");
  //for (auto t : results.times) {
  //  printf("\t%.1e", t);
  //}
  //printf("\n");

  // print metrics
  printf("%s%s%lu%s%s", name.c_str(), sep.c_str(), bytes, sep.c_str(), results.unit.c_str());
  for (auto m : results.metrics) {
    printf("%s%.1e", sep.c_str(), m);
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
  ("m,megabytes",
   "Number of megabytes",
    cxxopts::value<double>()->default_value("2048.0")) // 2G
  ("g,gigabytes",
   "Number of gigabytes",
    cxxopts::value<double>()->default_value("2.0")) // 2G
  ("s,seperator",
    "Seperator to use for output",
     cxxopts::value<std::string>()->default_value(","))
  ("h,help",
   "Show help")
  ;
  auto result = options.parse(argc, argv);

  bool help = result["help"].as<bool>();
  if (help) {
    printf("%s\n", options.help().c_str());
    exit(0);
  }

  size_t bytes = 0;
  if (result["gigabytes"].count()) {
    bytes = result["gigabytes"].as<double>() * (1024ul * 1024ul * 1024ul);
  }
  else if (result["megabytes"].count()) {
    bytes = result["megabytes"].as<double>() * (1024ul * 1024ul);
  } 
  else {
    bytes = result["bytes"].as<size_t>();
  }

  size_t numIters = result["num-iters"].as<size_t>();
  std::string seperator = result["seperator"].as<std::string>();

  print_header(numIters, seperator);

  Results results;

  results = triad<int, CUDAMallocManaged<int>>(bytes, 1, numIters);
  print("triad/cudamallocmanaged", bytes, results, seperator);

  results = triad_footprint<int, CUDAMallocManaged<int>>(bytes, 1, numIters);
  print("triad_footprint/cudamallocmanaged", bytes, results, seperator);

  results = triad<int, std::allocator<int>>(bytes, 1, numIters);
  print("triad/system", bytes, results, seperator);

  results = triad_footprint<int, std::allocator<int>>(bytes, 1, numIters);
  print("triad_footprint/system", bytes, results, seperator);

}
