#pragma once

#include <vector>
#include <string>

typedef struct {
  std::vector<double> times;
  std::vector<double> metrics;
  std::string unit;
} Results;