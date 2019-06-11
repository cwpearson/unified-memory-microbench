#pragma once

#include <cstdint>

inline size_t alignment_of(void *ptr) {
  for (size_t i = 2; i <= 512; i *= 2) {
    if ((uintptr_t(ptr) % i) != 0) {
      return i/2;
    }
  }
  return 512;
}

template <typename T>
T* align(T *ptr, size_t align) {

  uintptr_t u = uintptr_t(ptr);

  while (u % align != 0) {u++;}

  return (T*)u;
}
