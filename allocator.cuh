#include "cuda_runtime.cuh"

T *alloc_cuda_malloc_managed(const size_t n) {
    T *ptr= nullptr;
    CUDA_RUNTIME(cudaMallocManaged(&ptr, n * sizeof(T)));
    return ptr;
}

T *alloc_system(const size_t n) {
    T *ptr = (T*) malloc(n * sizeof(T));
    return ptr;
}