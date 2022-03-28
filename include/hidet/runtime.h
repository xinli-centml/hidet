#pragma once

#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>

#define CUDA_CALL(func) {                                           \
    cudaError_t e = (func);                                         \
    if(e != cudaSuccess) {                                          \
        std::cerr << __FILE__ << ": " << __LINE__ << ":"            \
        << "CUDA: " << cudaGetErrorString(e) << std::endl;          \
    }}

#define CUBLAS_CALL(func) {                                     \
    cublasStatus_t e = (func);                                  \
    if(e != CUBLAS_STATUS_SUCCESS) {                            \
        std::cerr << __FILE__ << ": " << __LINE__ << ":"        \
        << "CUBLAS: error code " << e << std::endl;             \
    }}

#ifdef assert
#undef assert
#endif
#define assert(x) if(!(x)){                                         \
        std::cerr << __FILE__ << ": " << __LINE__ << ": "           \
        << #x << " failed" << std::endl;                            \
}

#ifndef DLL
#define DLL extern "C" __attribute__((visibility("default")))
#endif
