#pragma once

#include "check_cuda.cuh"

#include <limits>

template <class T> class CUDAMallocManaged {
public:
  // type definitions
  typedef T value_type;
  typedef T *pointer;
  typedef const T *const_pointer;
  typedef T &reference;
  typedef const T &const_reference;
  typedef std::size_t size_type;
  typedef std::ptrdiff_t difference_type;

  // rebind allocator to type U
  template <class U> struct rebind { typedef CUDAMallocManaged<U> other; };

  /* constructors and destructor
   * - nothing to do because the allocator has no state
   */
  CUDAMallocManaged() noexcept {}
  CUDAMallocManaged(const CUDAMallocManaged &) noexcept {}
  template <class U> CUDAMallocManaged(const CUDAMallocManaged<U> &) noexcept {}
  ~CUDAMallocManaged() noexcept {}

  // return maximum number of elements that can be allocated
  size_type max_size() const noexcept {
    return std::numeric_limits<std::size_t>::max() / sizeof(T);
  }

  // allocate but don't initialize num elements of type T
  pointer allocate(size_type num, const void * = 0) {
    pointer ret = nullptr;
    CUDA_RUNTIME(cudaMallocManaged(&ret, num * sizeof(T)));
    return ret;
  }

  // deallocate storage p of deleted elements
  void deallocate(pointer p, size_type num) {
      CUDA_RUNTIME(cudaFree(p));
  }
};

// return that all specializations of this allocator are interchangeable
template <class T1, class T2>
bool operator==(const CUDAMallocManaged<T1> &, const CUDAMallocManaged<T2> &) noexcept {
  return true;
}
template <class T1, class T2>
bool operator!=(const CUDAMallocManaged<T1> &, const CUDAMallocManaged<T2> &) noexcept {
  return false;
}
