#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <helper_cuda.h>

#include <nsparse.h>
#include <nsparse_asm.h>

__global__ void kernel_spmv_init_ans(real *d_ans,
                                     int M) {
  
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) {
        return;
    }
    d_ans[i] = 0;

}


__global__ void kernel_spmv_amb_atomic1(real *ans,
                                        real *value, unsigned short *col,
                                        const unsigned int* __restrict__ cl,
                                        const int* __restrict__ cs,
                                        const real* __restrict__ vector,
                                        unsigned short *d_permutation,
                                        const unsigned short* __restrict__ d_permutation_offset,
                                        int row_num,
                                        int seg_size) {
  
    int i = blockIdx.x * blockDim.x + threadIdx.x;
  
    if (i >= row_num) {
        return;
    }
  
    int c_index = i >> WARP_BIT;
    int offset = ld_gbl_ushort(d_permutation + i) + d_permutation_offset[c_index] * USHORT_MAX;
  
    int start = cs[c_index] + (threadIdx.x & (WARP - 1));
    int colstart = (cs[c_index] / 1) + (threadIdx.x & (WARP - 1));
    int length = cl[c_index];
    int width = length & SCL_BIT;
    int c_offset = (length >> SCL_BORDER) * seg_size;
  
    int h;
    int c = ld_gbl_ushort(col + colstart) + c_offset;
    real answer = ld_gbl_val(value + start) * vector[c];
    start += WARP;
    colstart += WARP;

    for (h = 0; h < width; h++) {
        c = ld_gbl_ushort(col + colstart) + c_offset;
        answer += ld_gbl_val(value + start) * vector[c];
        start += WARP;
        colstart += WARP;
    }
  
#ifdef FLOAT
    atomicAdd(ans + offset, answer);
#else
    unsigned long long int *address_ull = (unsigned long long int *)(ans + offset);
    unsigned long long int old = *address_ull;
    unsigned long long int assumed;
    do {
        assumed = old;
        old = atomicCAS(address_ull, assumed, __double_as_longlong(answer + __longlong_as_double(assumed)));
    } while (assumed != old);
  
#endif
}

__global__ void kernel_spmv_amb_atomic2(real *ans,
                                        real *value, unsigned short *col,
                                        const unsigned int* __restrict__ cl,
                                        const int* __restrict__ cs,
                                        const real* __restrict__ vector,
                                        unsigned short *d_permutation,
                                        const unsigned short* __restrict__ d_permutation_offset,
                                        int row_num,
                                        int seg_size) {
  
    int i = blockIdx.x * blockDim.x + threadIdx.x;
  
    if (i >= row_num) {
        return;
    }
  
    int c_index = i >> WARP_BIT;
    int offset = ld_gbl_ushort(d_permutation + i) + d_permutation_offset[c_index] * USHORT_MAX;
  
    int start = cs[c_index] + (threadIdx.x & (WARP - 1));
    int colstart = (cs[c_index] / 2) + (threadIdx.x & (WARP - 1));
    int length = cl[c_index];
    int width = length & SCL_BIT;
    int c_offset = (length >> SCL_BORDER) * seg_size;
  
    int h;
    int c = ld_gbl_ushort(col + colstart) + c_offset;
    real answer = ld_gbl_val(value + start) * vector[c];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 1];
    start += WARP;
    colstart += WARP;

    for (h = 0; h < width; h++) {
        c = ld_gbl_ushort(col + colstart) + c_offset;
        answer += ld_gbl_val(value + start) * vector[c];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 1];
        start += WARP;
        colstart += WARP;
    }
  
#ifdef FLOAT
    atomicAdd(ans + offset, answer);
#else
    unsigned long long int *address_ull = (unsigned long long int *)(ans + offset);
    unsigned long long int old = *address_ull;
    unsigned long long int assumed;
    do {
        assumed = old;
        old = atomicCAS(address_ull, assumed, __double_as_longlong(answer + __longlong_as_double(assumed)));
    } while (assumed != old);
  
#endif
}

__global__ void kernel_spmv_amb_atomic3(real *ans,
                                        real *value, unsigned short *col,
                                        const unsigned int* __restrict__ cl,
                                        const int* __restrict__ cs,
                                        const real* __restrict__ vector,
                                        unsigned short *d_permutation,
                                        const unsigned short* __restrict__ d_permutation_offset,
                                        int row_num,
                                        int seg_size) {
  
    int i = blockIdx.x * blockDim.x + threadIdx.x;
  
    if (i >= row_num) {
        return;
    }
  
    int c_index = i >> WARP_BIT;
    int offset = ld_gbl_ushort(d_permutation + i) + d_permutation_offset[c_index] * USHORT_MAX;
  
    int start = cs[c_index] + (threadIdx.x & (WARP - 1));
    int colstart = (cs[c_index] / 3) + (threadIdx.x & (WARP - 1));
    int length = cl[c_index];
    int width = length & SCL_BIT;
    int c_offset = (length >> SCL_BORDER) * seg_size;
  
    int h;
    int c = ld_gbl_ushort(col + colstart) + c_offset;
    real answer = ld_gbl_val(value + start) * vector[c];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 1];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 2];
    start += WARP;
    colstart += WARP;

    for (h = 0; h < width; h++) {
        c = ld_gbl_ushort(col + colstart) + c_offset;
        answer += ld_gbl_val(value + start) * vector[c];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 1];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 2];
        start += WARP;
        colstart += WARP;
    }
  
#ifdef FLOAT
    atomicAdd(ans + offset, answer);
#else
    unsigned long long int *address_ull = (unsigned long long int *)(ans + offset);
    unsigned long long int old = *address_ull;
    unsigned long long int assumed;
    do {
        assumed = old;
        old = atomicCAS(address_ull, assumed, __double_as_longlong(answer + __longlong_as_double(assumed)));
    } while (assumed != old);
  
#endif
}

__global__ void kernel_spmv_amb_atomic4(real *ans,
                                        real *value, unsigned short *col,
                                        const unsigned int* __restrict__ cl,
                                        const int* __restrict__ cs,
                                        const real* __restrict__ vector,
                                        unsigned short *d_permutation,
                                        const unsigned short* __restrict__ d_permutation_offset,
                                        int row_num,
                                        int seg_size) {
  
    int i = blockIdx.x * blockDim.x + threadIdx.x;
  
    if (i >= row_num) {
        return;
    }
  
    int c_index = i >> WARP_BIT;
    int offset = ld_gbl_ushort(d_permutation + i) + d_permutation_offset[c_index] * USHORT_MAX;
  
    int start = cs[c_index] + (threadIdx.x & (WARP - 1));
    int colstart = (cs[c_index] / 4) + (threadIdx.x & (WARP - 1));
    int length = cl[c_index];
    int width = length & SCL_BIT;
    int c_offset = (length >> SCL_BORDER) * seg_size;
  
    int h;
    int c = ld_gbl_ushort(col + colstart) + c_offset;
    real answer = ld_gbl_val(value + start) * vector[c];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 1];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 2];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 3];
    start += WARP;
    colstart += WARP;

    for (h = 0; h < width; h++) {
        c = ld_gbl_ushort(col + colstart) + c_offset;
        answer += ld_gbl_val(value + start) * vector[c];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 1];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 2];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 3];
        start += WARP;
        colstart += WARP;
    }
  
#ifdef FLOAT
    atomicAdd(ans + offset, answer);
#else
    unsigned long long int *address_ull = (unsigned long long int *)(ans + offset);
    unsigned long long int old = *address_ull;
    unsigned long long int assumed;
    do {
        assumed = old;
        old = atomicCAS(address_ull, assumed, __double_as_longlong(answer + __longlong_as_double(assumed)));
    } while (assumed != old);
  
#endif
}

__global__ void kernel_spmv_amb_atomic5(real *ans,
                                        real *value, unsigned short *col,
                                        const unsigned int* __restrict__ cl,
                                        const int* __restrict__ cs,
                                        const real* __restrict__ vector,
                                        unsigned short *d_permutation,
                                        const unsigned short* __restrict__ d_permutation_offset,
                                        int row_num,
                                        int seg_size) {
  
    int i = blockIdx.x * blockDim.x + threadIdx.x;
  
    if (i >= row_num) {
        return;
    }
  
    int c_index = i >> WARP_BIT;
    int offset = ld_gbl_ushort(d_permutation + i) + d_permutation_offset[c_index] * USHORT_MAX;
  
    int start = cs[c_index] + (threadIdx.x & (WARP - 1));
    int colstart = (cs[c_index] / 5) + (threadIdx.x & (WARP - 1));
    int length = cl[c_index];
    int width = length & SCL_BIT;
    int c_offset = (length >> SCL_BORDER) * seg_size;
  
    int h;
    int c = ld_gbl_ushort(col + colstart) + c_offset;
    real answer = ld_gbl_val(value + start) * vector[c];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 1];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 2];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 3];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 4];
    start += WARP;
    colstart += WARP;

    for (h = 0; h < width; h++) {
        c = ld_gbl_ushort(col + colstart) + c_offset;
        answer += ld_gbl_val(value + start) * vector[c];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 1];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 2];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 3];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 4];
        start += WARP;
        colstart += WARP;
    }
  
#ifdef FLOAT
    atomicAdd(ans + offset, answer);
#else
    unsigned long long int *address_ull = (unsigned long long int *)(ans + offset);
    unsigned long long int old = *address_ull;
    unsigned long long int assumed;
    do {
        assumed = old;
        old = atomicCAS(address_ull, assumed, __double_as_longlong(answer + __longlong_as_double(assumed)));
    } while (assumed != old);
  
#endif
}

__global__ void kernel_spmv_amb_atomic6(real *ans,
                                        real *value, unsigned short *col,
                                        const unsigned int* __restrict__ cl,
                                        const int* __restrict__ cs,
                                        const real* __restrict__ vector,
                                        unsigned short *d_permutation,
                                        const unsigned short* __restrict__ d_permutation_offset,
                                        int row_num,
                                        int seg_size) {
  
    int i = blockIdx.x * blockDim.x + threadIdx.x;
  
    if (i >= row_num) {
        return;
    }
  
    int c_index = i >> WARP_BIT;
    int offset = ld_gbl_ushort(d_permutation + i) + d_permutation_offset[c_index] * USHORT_MAX;
  
    int start = cs[c_index] + (threadIdx.x & (WARP - 1));
    int colstart = (cs[c_index] / 6) + (threadIdx.x & (WARP - 1));
    int length = cl[c_index];
    int width = length & SCL_BIT;
    int c_offset = (length >> SCL_BORDER) * seg_size;
  
    int h;
    int c = ld_gbl_ushort(col + colstart) + c_offset;
    real answer = ld_gbl_val(value + start) * vector[c];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 1];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 2];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 3];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 4];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 5];
    start += WARP;
    colstart += WARP;

    for (h = 0; h < width; h++) {
        c = ld_gbl_ushort(col + colstart) + c_offset;
        answer += ld_gbl_val(value + start) * vector[c];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 1];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 2];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 3];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 4];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 5];
        start += WARP;
        colstart += WARP;
    }
  
#ifdef FLOAT
    atomicAdd(ans + offset, answer);
#else
    unsigned long long int *address_ull = (unsigned long long int *)(ans + offset);
    unsigned long long int old = *address_ull;
    unsigned long long int assumed;
    do {
        assumed = old;
        old = atomicCAS(address_ull, assumed, __double_as_longlong(answer + __longlong_as_double(assumed)));
    } while (assumed != old);
  
#endif
}

__global__ void kernel_spmv_amb_atomic7(real *ans,
                                        real *value, unsigned short *col,
                                        const unsigned int* __restrict__ cl,
                                        const int* __restrict__ cs,
                                        const real* __restrict__ vector,
                                        unsigned short *d_permutation,
                                        const unsigned short* __restrict__ d_permutation_offset,
                                        int row_num,
                                        int seg_size) {
  
    int i = blockIdx.x * blockDim.x + threadIdx.x;
  
    if (i >= row_num) {
        return;
    }
  
    int c_index = i >> WARP_BIT;
    int offset = ld_gbl_ushort(d_permutation + i) + d_permutation_offset[c_index] * USHORT_MAX;
  
    int start = cs[c_index] + (threadIdx.x & (WARP - 1));
    int colstart = (cs[c_index] / 7) + (threadIdx.x & (WARP - 1));
    int length = cl[c_index];
    int width = length & SCL_BIT;
    int c_offset = (length >> SCL_BORDER) * seg_size;
  
    int h;
    int c = ld_gbl_ushort(col + colstart) + c_offset;
    real answer = ld_gbl_val(value + start) * vector[c];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 1];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 2];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 3];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 4];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 5];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 6];
    start += WARP;
    colstart += WARP;

    for (h = 0; h < width; h++) {
        c = ld_gbl_ushort(col + colstart) + c_offset;
        answer += ld_gbl_val(value + start) * vector[c];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 1];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 2];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 3];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 4];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 5];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 6];
        start += WARP;
        colstart += WARP;
    }
  
#ifdef FLOAT
    atomicAdd(ans + offset, answer);
#else
    unsigned long long int *address_ull = (unsigned long long int *)(ans + offset);
    unsigned long long int old = *address_ull;
    unsigned long long int assumed;
    do {
        assumed = old;
        old = atomicCAS(address_ull, assumed, __double_as_longlong(answer + __longlong_as_double(assumed)));
    } while (assumed != old);
  
#endif
}

__global__ void kernel_spmv_amb_atomic8(real *ans,
                                        real *value, unsigned short *col,
                                        const unsigned int* __restrict__ cl,
                                        const int* __restrict__ cs,
                                        const real* __restrict__ vector,
                                        unsigned short *d_permutation,
                                        const unsigned short* __restrict__ d_permutation_offset,
                                        int row_num,
                                        int seg_size) {
  
    int i = blockIdx.x * blockDim.x + threadIdx.x;
  
    if (i >= row_num) {
        return;
    }
  
    int c_index = i >> WARP_BIT;
    int offset = ld_gbl_ushort(d_permutation + i) + d_permutation_offset[c_index] * USHORT_MAX;
  
    int start = cs[c_index] + (threadIdx.x & (WARP - 1));
    int colstart = (cs[c_index] / 8) + (threadIdx.x & (WARP - 1));
    int length = cl[c_index];
    int width = length & SCL_BIT;
    int c_offset = (length >> SCL_BORDER) * seg_size;
  
    int h;
    int c = ld_gbl_ushort(col + colstart) + c_offset;
    real answer = ld_gbl_val(value + start) * vector[c];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 1];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 2];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 3];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 4];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 5];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 6];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 7];
    start += WARP;
    colstart += WARP;

    for (h = 0; h < width; h++) {
        c = ld_gbl_ushort(col + colstart) + c_offset;
        answer += ld_gbl_val(value + start) * vector[c];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 1];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 2];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 3];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 4];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 5];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 6];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 7];
        start += WARP;
        colstart += WARP;
    }
  
#ifdef FLOAT
    atomicAdd(ans + offset, answer);
#else
    unsigned long long int *address_ull = (unsigned long long int *)(ans + offset);
    unsigned long long int old = *address_ull;
    unsigned long long int assumed;
    do {
        assumed = old;
        old = atomicCAS(address_ull, assumed, __double_as_longlong(answer + __longlong_as_double(assumed)));
    } while (assumed != old);
  
#endif
}

__global__ void kernel_spmv_amb_atomic9(real *ans,
                                        real *value, unsigned short *col,
                                        const unsigned int* __restrict__ cl,
                                        const int* __restrict__ cs,
                                        const real* __restrict__ vector,
                                        unsigned short *d_permutation,
                                        const unsigned short* __restrict__ d_permutation_offset,
                                        int row_num,
                                        int seg_size) {
  
    int i = blockIdx.x * blockDim.x + threadIdx.x;
  
    if (i >= row_num) {
        return;
    }
  
    int c_index = i >> WARP_BIT;
    int offset = ld_gbl_ushort(d_permutation + i) + d_permutation_offset[c_index] * USHORT_MAX;
  
    int start = cs[c_index] + (threadIdx.x & (WARP - 1));
    int colstart = (cs[c_index] / 9) + (threadIdx.x & (WARP - 1));
    int length = cl[c_index];
    int width = length & SCL_BIT;
    int c_offset = (length >> SCL_BORDER) * seg_size;
  
    int h;
    int c = ld_gbl_ushort(col + colstart) + c_offset;
    real answer = ld_gbl_val(value + start) * vector[c];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 1];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 2];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 3];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 4];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 5];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 6];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 7];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 8];
    start += WARP;
    colstart += WARP;

    for (h = 0; h < width; h++) {
        c = ld_gbl_ushort(col + colstart) + c_offset;
        answer += ld_gbl_val(value + start) * vector[c];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 1];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 2];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 3];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 4];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 5];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 6];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 7];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 8];
        start += WARP;
        colstart += WARP;
    }
  
#ifdef FLOAT
    atomicAdd(ans + offset, answer);
#else
    unsigned long long int *address_ull = (unsigned long long int *)(ans + offset);
    unsigned long long int old = *address_ull;
    unsigned long long int assumed;
    do {
        assumed = old;
        old = atomicCAS(address_ull, assumed, __double_as_longlong(answer + __longlong_as_double(assumed)));
    } while (assumed != old);
  
#endif
}

__global__ void kernel_spmv_amb_atomic10(real *ans,
                                         real *value, unsigned short *col,
                                         const unsigned int* __restrict__ cl,
                                         const int* __restrict__ cs,
                                         const real* __restrict__ vector,
                                         unsigned short *d_permutation,
                                         const unsigned short* __restrict__ d_permutation_offset,
                                         int row_num,
                                         int seg_size) {
  
    int i = blockIdx.x * blockDim.x + threadIdx.x;
  
    if (i >= row_num) {
        return;
    }
  
    int c_index = i >> WARP_BIT;
    int offset = ld_gbl_ushort(d_permutation + i) + d_permutation_offset[c_index] * USHORT_MAX;
  
    int start = cs[c_index] + (threadIdx.x & (WARP - 1));
    int colstart = (cs[c_index] / 10) + (threadIdx.x & (WARP - 1));
    int length = cl[c_index];
    int width = length & SCL_BIT;
    int c_offset = (length >> SCL_BORDER) * seg_size;
  
    int h;
    int c = ld_gbl_ushort(col + colstart) + c_offset;
    real answer = ld_gbl_val(value + start) * vector[c];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 1];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 2];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 3];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 4];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 5];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 6];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 7];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 8];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 9];
    start += WARP;
    colstart += WARP;

    for (h = 0; h < width; h++) {
        c = ld_gbl_ushort(col + colstart) + c_offset;
        answer += ld_gbl_val(value + start) * vector[c];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 1];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 2];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 3];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 4];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 5];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 6];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 7];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 8];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 9];
        start += WARP;
        colstart += WARP;
    }
  
#ifdef FLOAT
    atomicAdd(ans + offset, answer);
#else
    unsigned long long int *address_ull = (unsigned long long int *)(ans + offset);
    unsigned long long int old = *address_ull;
    unsigned long long int assumed;
    do {
        assumed = old;
        old = atomicCAS(address_ull, assumed, __double_as_longlong(answer + __longlong_as_double(assumed)));
    } while (assumed != old);
  
#endif
}

__global__ void kernel_spmv_amb_atomic11(real *ans,
                                         real *value, unsigned short *col,
                                         const unsigned int* __restrict__ cl,
                                         const int* __restrict__ cs,
                                         const real* __restrict__ vector,
                                         unsigned short *d_permutation,
                                         const unsigned short* __restrict__ d_permutation_offset,
                                         int row_num,
                                         int seg_size) {
  
    int i = blockIdx.x * blockDim.x + threadIdx.x;
  
    if (i >= row_num) {
        return;
    }
  
    int c_index = i >> WARP_BIT;
    int offset = ld_gbl_ushort(d_permutation + i) + d_permutation_offset[c_index] * USHORT_MAX;
  
    int start = cs[c_index] + (threadIdx.x & (WARP - 1));
    int colstart = (cs[c_index] / 11) + (threadIdx.x & (WARP - 1));
    int length = cl[c_index];
    int width = length & SCL_BIT;
    int c_offset = (length >> SCL_BORDER) * seg_size;
  
    int h;
    int c = ld_gbl_ushort(col + colstart) + c_offset;
    real answer = ld_gbl_val(value + start) * vector[c];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 1];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 2];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 3];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 4];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 5];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 6];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 7];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 8];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 9];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 10];
    start += WARP;
    colstart += WARP;

    for (h = 0; h < width; h++) {
        c = ld_gbl_ushort(col + colstart) + c_offset;
        answer += ld_gbl_val(value + start) * vector[c];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 1];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 2];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 3];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 4];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 5];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 6];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 7];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 8];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 9];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 10];
        start += WARP;
        colstart += WARP;
    }
  
#ifdef FLOAT
    atomicAdd(ans + offset, answer);
#else
    unsigned long long int *address_ull = (unsigned long long int *)(ans + offset);
    unsigned long long int old = *address_ull;
    unsigned long long int assumed;
    do {
        assumed = old;
        old = atomicCAS(address_ull, assumed, __double_as_longlong(answer + __longlong_as_double(assumed)));
    } while (assumed != old);
  
#endif
}

__global__ void kernel_spmv_amb_atomic12(real *ans,
                                         real *value, unsigned short *col,
                                         const unsigned int* __restrict__ cl,
                                         const int* __restrict__ cs,
                                         const real* __restrict__ vector,
                                         unsigned short *d_permutation,
                                         const unsigned short* __restrict__ d_permutation_offset,
                                         int row_num,
                                         int seg_size) {
  
    int i = blockIdx.x * blockDim.x + threadIdx.x;
  
    if (i >= row_num) {
        return;
    }
  
    int c_index = i >> WARP_BIT;
    int offset = ld_gbl_ushort(d_permutation + i) + d_permutation_offset[c_index] * USHORT_MAX;
  
    int start = cs[c_index] + (threadIdx.x & (WARP - 1));
    int colstart = (cs[c_index] / 12) + (threadIdx.x & (WARP - 1));
    int length = cl[c_index];
    int width = length & SCL_BIT;
    int c_offset = (length >> SCL_BORDER) * seg_size;
  
    int h;
    int c = ld_gbl_ushort(col + colstart) + c_offset;
    real answer = ld_gbl_val(value + start) * vector[c];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 1];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 2];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 3];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 4];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 5];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 6];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 7];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 8];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 9];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 10];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 11];
    start += WARP;
    colstart += WARP;

    for (h = 0; h < width; h++) {
        c = ld_gbl_ushort(col + colstart) + c_offset;
        answer += ld_gbl_val(value + start) * vector[c];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 1];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 2];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 3];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 4];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 5];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 6];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 7];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 8];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 9];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 10];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 11];
        start += WARP;
        colstart += WARP;
    }
  
#ifdef FLOAT
    atomicAdd(ans + offset, answer);
#else
    unsigned long long int *address_ull = (unsigned long long int *)(ans + offset);
    unsigned long long int old = *address_ull;
    unsigned long long int assumed;
    do {
        assumed = old;
        old = atomicCAS(address_ull, assumed, __double_as_longlong(answer + __longlong_as_double(assumed)));
    } while (assumed != old);
  
#endif
}

__global__ void kernel_spmv_amb_atomic13(real *ans,
                                         real *value, unsigned short *col,
                                         const unsigned int* __restrict__ cl,
                                         const int* __restrict__ cs,
                                         const real* __restrict__ vector,
                                         unsigned short *d_permutation,
                                         const unsigned short* __restrict__ d_permutation_offset,
                                         int row_num,
                                         int seg_size) {
  
    int i = blockIdx.x * blockDim.x + threadIdx.x;
  
    if (i >= row_num) {
        return;
    }
  
    int c_index = i >> WARP_BIT;
    int offset = ld_gbl_ushort(d_permutation + i) + d_permutation_offset[c_index] * USHORT_MAX;
  
    int start = cs[c_index] + (threadIdx.x & (WARP - 1));
    int colstart = (cs[c_index] / 13) + (threadIdx.x & (WARP - 1));
    int length = cl[c_index];
    int width = length & SCL_BIT;
    int c_offset = (length >> SCL_BORDER) * seg_size;
  
    int h;
    int c = ld_gbl_ushort(col + colstart) + c_offset;
    real answer = ld_gbl_val(value + start) * vector[c];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 1];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 2];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 3];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 4];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 5];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 6];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 7];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 8];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 9];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 10];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 11];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 12];
    start += WARP;
    colstart += WARP;

    for (h = 0; h < width; h++) {
        c = ld_gbl_ushort(col + colstart) + c_offset;
        answer += ld_gbl_val(value + start) * vector[c];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 1];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 2];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 3];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 4];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 5];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 6];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 7];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 8];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 9];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 10];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 11];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 12];
        start += WARP;
        colstart += WARP;
    }
  
#ifdef FLOAT
    atomicAdd(ans + offset, answer);
#else
    unsigned long long int *address_ull = (unsigned long long int *)(ans + offset);
    unsigned long long int old = *address_ull;
    unsigned long long int assumed;
    do {
        assumed = old;
        old = atomicCAS(address_ull, assumed, __double_as_longlong(answer + __longlong_as_double(assumed)));
    } while (assumed != old);
  
#endif
}

__global__ void kernel_spmv_amb_atomic14(real *ans,
                                         real *value, unsigned short *col,
                                         const unsigned int* __restrict__ cl,
                                         const int* __restrict__ cs,
                                         const real* __restrict__ vector,
                                         unsigned short *d_permutation,
                                         const unsigned short* __restrict__ d_permutation_offset,
                                         int row_num,
                                         int seg_size) {
  
    int i = blockIdx.x * blockDim.x + threadIdx.x;
  
    if (i >= row_num) {
        return;
    }
  
    int c_index = i >> WARP_BIT;
    int offset = ld_gbl_ushort(d_permutation + i) + d_permutation_offset[c_index] * USHORT_MAX;
  
    int start = cs[c_index] + (threadIdx.x & (WARP - 1));
    int colstart = (cs[c_index] / 14) + (threadIdx.x & (WARP - 1));
    int length = cl[c_index];
    int width = length & SCL_BIT;
    int c_offset = (length >> SCL_BORDER) * seg_size;
  
    int h;
    int c = ld_gbl_ushort(col + colstart) + c_offset;
    real answer = ld_gbl_val(value + start) * vector[c];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 1];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 2];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 3];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 4];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 5];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 6];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 7];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 8];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 9];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 10];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 11];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 12];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 13];
    start += WARP;
    colstart += WARP;

    for (h = 0; h < width; h++) {
        c = ld_gbl_ushort(col + colstart) + c_offset;
        answer += ld_gbl_val(value + start) * vector[c];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 1];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 2];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 3];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 4];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 5];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 6];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 7];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 8];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 9];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 10];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 11];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 12];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 13];
        start += WARP;
        colstart += WARP;
    }
  
#ifdef FLOAT
    atomicAdd(ans + offset, answer);
#else
    unsigned long long int *address_ull = (unsigned long long int *)(ans + offset);
    unsigned long long int old = *address_ull;
    unsigned long long int assumed;
    do {
        assumed = old;
        old = atomicCAS(address_ull, assumed, __double_as_longlong(answer + __longlong_as_double(assumed)));
    } while (assumed != old);
  
#endif
}

__global__ void kernel_spmv_amb_atomic15(real *ans,
                                         real *value, unsigned short *col,
                                         const unsigned int* __restrict__ cl,
                                         const int* __restrict__ cs,
                                         const real* __restrict__ vector,
                                         unsigned short *d_permutation,
                                         const unsigned short* __restrict__ d_permutation_offset,
                                         int row_num,
                                         int seg_size) {
  
    int i = blockIdx.x * blockDim.x + threadIdx.x;
  
    if (i >= row_num) {
        return;
    }
  
    int c_index = i >> WARP_BIT;
    int offset = ld_gbl_ushort(d_permutation + i) + d_permutation_offset[c_index] * USHORT_MAX;
  
    int start = cs[c_index] + (threadIdx.x & (WARP - 1));
    int colstart = (cs[c_index] / 15) + (threadIdx.x & (WARP - 1));
    int length = cl[c_index];
    int width = length & SCL_BIT;
    int c_offset = (length >> SCL_BORDER) * seg_size;
  
    int h;
    int c = ld_gbl_ushort(col + colstart) + c_offset;
    real answer = ld_gbl_val(value + start) * vector[c];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 1];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 2];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 3];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 4];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 5];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 6];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 7];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 8];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 9];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 10];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 11];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 12];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 13];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 14];
    start += WARP;
    colstart += WARP;

    for (h = 0; h < width; h++) {
        c = ld_gbl_ushort(col + colstart) + c_offset;
        answer += ld_gbl_val(value + start) * vector[c];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 1];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 2];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 3];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 4];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 5];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 6];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 7];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 8];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 9];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 10];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 11];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 12];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 13];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 14];
        start += WARP;
        colstart += WARP;
    }
  
#ifdef FLOAT
    atomicAdd(ans + offset, answer);
#else
    unsigned long long int *address_ull = (unsigned long long int *)(ans + offset);
    unsigned long long int old = *address_ull;
    unsigned long long int assumed;
    do {
        assumed = old;
        old = atomicCAS(address_ull, assumed, __double_as_longlong(answer + __longlong_as_double(assumed)));
    } while (assumed != old);
  
#endif
}

__global__ void kernel_spmv_amb_atomic16(real *ans,
                                         real *value, unsigned short *col,
                                         const unsigned int* __restrict__ cl,
                                         const int* __restrict__ cs,
                                         const real* __restrict__ vector,
                                         unsigned short *d_permutation,
                                         const unsigned short* __restrict__ d_permutation_offset,
                                         int row_num,
                                         int seg_size) {
  
    int i = blockIdx.x * blockDim.x + threadIdx.x;
  
    if (i >= row_num) {
        return;
    }
  
    int c_index = i >> WARP_BIT;
    int offset = ld_gbl_ushort(d_permutation + i) + d_permutation_offset[c_index] * USHORT_MAX;
  
    int start = cs[c_index] + (threadIdx.x & (WARP - 1));
    int colstart = (cs[c_index] / 16) + (threadIdx.x & (WARP - 1));
    int length = cl[c_index];
    int width = length & SCL_BIT;
    int c_offset = (length >> SCL_BORDER) * seg_size;
  
    int h;
    int c = ld_gbl_ushort(col + colstart) + c_offset;
    real answer = ld_gbl_val(value + start) * vector[c];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 1];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 2];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 3];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 4];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 5];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 6];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 7];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 8];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 9];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 10];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 11];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 12];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 13];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 14];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 15];
    start += WARP;
    colstart += WARP;

    for (h = 0; h < width; h++) {
        c = ld_gbl_ushort(col + colstart) + c_offset;
        answer += ld_gbl_val(value + start) * vector[c];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 1];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 2];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 3];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 4];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 5];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 6];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 7];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 8];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 9];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 10];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 11];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 12];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 13];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 14];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 15];
        start += WARP;
        colstart += WARP;
    }
  
#ifdef FLOAT
    atomicAdd(ans + offset, answer);
#else
    unsigned long long int *address_ull = (unsigned long long int *)(ans + offset);
    unsigned long long int old = *address_ull;
    unsigned long long int assumed;
    do {
        assumed = old;
        old = atomicCAS(address_ull, assumed, __double_as_longlong(answer + __longlong_as_double(assumed)));
    } while (assumed != old);
  
#endif
}

__global__ void kernel_spmv_amb_atomic17(real *ans,
                                         real *value, unsigned short *col,
                                         const unsigned int* __restrict__ cl,
                                         const int* __restrict__ cs,
                                         const real* __restrict__ vector,
                                         unsigned short *d_permutation,
                                         const unsigned short* __restrict__ d_permutation_offset,
                                         int row_num,
                                         int seg_size) {
  
    int i = blockIdx.x * blockDim.x + threadIdx.x;
  
    if (i >= row_num) {
        return;
    }
  
    int c_index = i >> WARP_BIT;
    int offset = ld_gbl_ushort(d_permutation + i) + d_permutation_offset[c_index] * USHORT_MAX;
  
    int start = cs[c_index] + (threadIdx.x & (WARP - 1));
    int colstart = (cs[c_index] / 17) + (threadIdx.x & (WARP - 1));
    int length = cl[c_index];
    int width = length & SCL_BIT;
    int c_offset = (length >> SCL_BORDER) * seg_size;
  
    int h;
    int c = ld_gbl_ushort(col + colstart) + c_offset;
    real answer = ld_gbl_val(value + start) * vector[c];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 1];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 2];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 3];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 4];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 5];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 6];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 7];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 8];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 9];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 10];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 11];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 12];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 13];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 14];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 15];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 16];
    start += WARP;
    colstart += WARP;

    for (h = 0; h < width; h++) {
        c = ld_gbl_ushort(col + colstart) + c_offset;
        answer += ld_gbl_val(value + start) * vector[c];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 1];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 2];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 3];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 4];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 5];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 6];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 7];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 8];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 9];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 10];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 11];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 12];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 13];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 14];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 15];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 16];
        start += WARP;
        colstart += WARP;
    }
  
#ifdef FLOAT
    atomicAdd(ans + offset, answer);
#else
    unsigned long long int *address_ull = (unsigned long long int *)(ans + offset);
    unsigned long long int old = *address_ull;
    unsigned long long int assumed;
    do {
        assumed = old;
        old = atomicCAS(address_ull, assumed, __double_as_longlong(answer + __longlong_as_double(assumed)));
    } while (assumed != old);
  
#endif
}

__global__ void kernel_spmv_amb_atomic18(real *ans,
                                         real *value, unsigned short *col,
                                         const unsigned int* __restrict__ cl,
                                         const int* __restrict__ cs,
                                         const real* __restrict__ vector,
                                         unsigned short *d_permutation,
                                         const unsigned short* __restrict__ d_permutation_offset,
                                         int row_num,
                                         int seg_size) {
  
    int i = blockIdx.x * blockDim.x + threadIdx.x;
  
    if (i >= row_num) {
        return;
    }
  
    int c_index = i >> WARP_BIT;
    int offset = ld_gbl_ushort(d_permutation + i) + d_permutation_offset[c_index] * USHORT_MAX;
  
    int start = cs[c_index] + (threadIdx.x & (WARP - 1));
    int colstart = (cs[c_index] / 18) + (threadIdx.x & (WARP - 1));
    int length = cl[c_index];
    int width = length & SCL_BIT;
    int c_offset = (length >> SCL_BORDER) * seg_size;
  
    int h;
    int c = ld_gbl_ushort(col + colstart) + c_offset;
    real answer = ld_gbl_val(value + start) * vector[c];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 1];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 2];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 3];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 4];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 5];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 6];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 7];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 8];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 9];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 10];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 11];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 12];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 13];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 14];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 15];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 16];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 17];
    start += WARP;
    colstart += WARP;

    for (h = 0; h < width; h++) {
        c = ld_gbl_ushort(col + colstart) + c_offset;
        answer += ld_gbl_val(value + start) * vector[c];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 1];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 2];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 3];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 4];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 5];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 6];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 7];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 8];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 9];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 10];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 11];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 12];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 13];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 14];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 15];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 16];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 17];
        start += WARP;
        colstart += WARP;
    }
  
#ifdef FLOAT
    atomicAdd(ans + offset, answer);
#else
    unsigned long long int *address_ull = (unsigned long long int *)(ans + offset);
    unsigned long long int old = *address_ull;
    unsigned long long int assumed;
    do {
        assumed = old;
        old = atomicCAS(address_ull, assumed, __double_as_longlong(answer + __longlong_as_double(assumed)));
    } while (assumed != old);
  
#endif
}

__global__ void kernel_spmv_amb_atomic19(real *ans,
                                         real *value, unsigned short *col,
                                         const unsigned int* __restrict__ cl,
                                         const int* __restrict__ cs,
                                         const real* __restrict__ vector,
                                         unsigned short *d_permutation,
                                         const unsigned short* __restrict__ d_permutation_offset,
                                         int row_num,
                                         int seg_size) {
  
    int i = blockIdx.x * blockDim.x + threadIdx.x;
  
    if (i >= row_num) {
        return;
    }
  
    int c_index = i >> WARP_BIT;
    int offset = ld_gbl_ushort(d_permutation + i) + d_permutation_offset[c_index] * USHORT_MAX;
  
    int start = cs[c_index] + (threadIdx.x & (WARP - 1));
    int colstart = (cs[c_index] / 19) + (threadIdx.x & (WARP - 1));
    int length = cl[c_index];
    int width = length & SCL_BIT;
    int c_offset = (length >> SCL_BORDER) * seg_size;
  
    int h;
    int c = ld_gbl_ushort(col + colstart) + c_offset;
    real answer = ld_gbl_val(value + start) * vector[c];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 1];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 2];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 3];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 4];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 5];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 6];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 7];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 8];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 9];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 10];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 11];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 12];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 13];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 14];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 15];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 16];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 17];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 18];
    start += WARP;
    colstart += WARP;

    for (h = 0; h < width; h++) {
        c = ld_gbl_ushort(col + colstart) + c_offset;
        answer += ld_gbl_val(value + start) * vector[c];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 1];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 2];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 3];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 4];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 5];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 6];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 7];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 8];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 9];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 10];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 11];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 12];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 13];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 14];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 15];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 16];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 17];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 18];
        start += WARP;
        colstart += WARP;
    }
  
#ifdef FLOAT
    atomicAdd(ans + offset, answer);
#else
    unsigned long long int *address_ull = (unsigned long long int *)(ans + offset);
    unsigned long long int old = *address_ull;
    unsigned long long int assumed;
    do {
        assumed = old;
        old = atomicCAS(address_ull, assumed, __double_as_longlong(answer + __longlong_as_double(assumed)));
    } while (assumed != old);
  
#endif
}

__global__ void kernel_spmv_amb_atomic20(real *ans,
                                         real *value, unsigned short *col,
                                         const unsigned int* __restrict__ cl,
                                         const int* __restrict__ cs,
                                         const real* __restrict__ vector,
                                         unsigned short *d_permutation,
                                         const unsigned short* __restrict__ d_permutation_offset,
                                         int row_num,
                                         int seg_size) {
  
    int i = blockIdx.x * blockDim.x + threadIdx.x;
  
    if (i >= row_num) {
        return;
    }
  
    int c_index = i >> WARP_BIT;
    int offset = ld_gbl_ushort(d_permutation + i) + d_permutation_offset[c_index] * USHORT_MAX;
  
    int start = cs[c_index] + (threadIdx.x & (WARP - 1));
    int colstart = (cs[c_index] / 20) + (threadIdx.x & (WARP - 1));
    int length = cl[c_index];
    int width = length & SCL_BIT;
    int c_offset = (length >> SCL_BORDER) * seg_size;
  
    int h;
    int c = ld_gbl_ushort(col + colstart) + c_offset;
    real answer = ld_gbl_val(value + start) * vector[c];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 1];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 2];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 3];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 4];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 5];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 6];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 7];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 8];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 9];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 10];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 11];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 12];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 13];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 14];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 15];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 16];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 17];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 18];
    start += WARP;
    answer += ld_gbl_val(value + start) * vector[c + 19];
    start += WARP;
    colstart += WARP;

    for (h = 0; h < width; h++) {
        c = ld_gbl_ushort(col + colstart) + c_offset;
        answer += ld_gbl_val(value + start) * vector[c];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 1];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 2];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 3];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 4];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 5];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 6];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 7];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 8];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 9];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 10];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 11];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 12];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 13];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 14];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 15];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 16];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 17];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 18];
        start += WARP;
        answer += ld_gbl_val(value + start) * vector[c + 19];
        start += WARP;
        colstart += WARP;
    }
  
#ifdef FLOAT
    atomicAdd(ans + offset, answer);
#else
    unsigned long long int *address_ull = (unsigned long long int *)(ans + offset);
    unsigned long long int old = *address_ull;
    unsigned long long int assumed;
    do {
        assumed = old;
        old = atomicCAS(address_ull, assumed, __double_as_longlong(answer + __longlong_as_double(assumed)));
    } while (assumed != old);
  
#endif
}

void sf_spmv_amb(real *d_y, sfAMB *mat, real *d_x, sfPlan *plan) {
  
    kernel_spmv_init_ans<<<div_round_up(mat->M, MAX_LOCAL_THREAD_NUM), MAX_LOCAL_THREAD_NUM>>>(d_y, mat->M);

    if (mat->block_size == 1) {
        kernel_spmv_amb_atomic1<<<plan->thread_grid, plan->thread_block>>>(d_y, mat->d_sellcs_val, mat->d_sellcs_col, mat->d_cl, mat->d_cs, d_x, mat->d_s_write_permutation, mat->d_s_write_permutation_offset, mat->c_size * mat->chunk, mat->seg_size);
    }

    else if (mat->block_size == 2) {
        kernel_spmv_amb_atomic2<<<plan->thread_grid, plan->thread_block>>>(d_y, mat->d_sellcs_val, mat->d_sellcs_col, mat->d_cl, mat->d_cs, d_x, mat->d_s_write_permutation, mat->d_s_write_permutation_offset, mat->c_size * mat->chunk, mat->seg_size);
    }
    else if (mat->block_size == 3) {
        kernel_spmv_amb_atomic3<<<plan->thread_grid, plan->thread_block>>>(d_y, mat->d_sellcs_val, mat->d_sellcs_col, mat->d_cl, mat->d_cs, d_x, mat->d_s_write_permutation, mat->d_s_write_permutation_offset, mat->c_size * mat->chunk, mat->seg_size);
    }
    else if (mat->block_size == 4) {
        kernel_spmv_amb_atomic4<<<plan->thread_grid, plan->thread_block>>>(d_y, mat->d_sellcs_val, mat->d_sellcs_col, mat->d_cl, mat->d_cs, d_x, mat->d_s_write_permutation, mat->d_s_write_permutation_offset, mat->c_size * mat->chunk, mat->seg_size);
    }
    else if (mat->block_size == 5) {
        kernel_spmv_amb_atomic5<<<plan->thread_grid, plan->thread_block>>>(d_y, mat->d_sellcs_val, mat->d_sellcs_col, mat->d_cl, mat->d_cs, d_x, mat->d_s_write_permutation, mat->d_s_write_permutation_offset, mat->c_size * mat->chunk, mat->seg_size);
    }
    else if (mat->block_size == 6) {
        kernel_spmv_amb_atomic6<<<plan->thread_grid, plan->thread_block>>>(d_y, mat->d_sellcs_val, mat->d_sellcs_col, mat->d_cl, mat->d_cs, d_x, mat->d_s_write_permutation, mat->d_s_write_permutation_offset, mat->c_size * mat->chunk, mat->seg_size);
    }
    else if (mat->block_size == 7) {
        kernel_spmv_amb_atomic7<<<plan->thread_grid, plan->thread_block>>>(d_y, mat->d_sellcs_val, mat->d_sellcs_col, mat->d_cl, mat->d_cs, d_x, mat->d_s_write_permutation, mat->d_s_write_permutation_offset, mat->c_size * mat->chunk, mat->seg_size);
    }
    else if (mat->block_size == 8) {
        kernel_spmv_amb_atomic8<<<plan->thread_grid, plan->thread_block>>>(d_y, mat->d_sellcs_val, mat->d_sellcs_col, mat->d_cl, mat->d_cs, d_x, mat->d_s_write_permutation, mat->d_s_write_permutation_offset, mat->c_size * mat->chunk, mat->seg_size);
    }
    else if (mat->block_size == 9) {
        kernel_spmv_amb_atomic9<<<plan->thread_grid, plan->thread_block>>>(d_y, mat->d_sellcs_val, mat->d_sellcs_col, mat->d_cl, mat->d_cs, d_x, mat->d_s_write_permutation, mat->d_s_write_permutation_offset, mat->c_size * mat->chunk, mat->seg_size);
    }
    else if (mat->block_size == 10) {
        kernel_spmv_amb_atomic10<<<plan->thread_grid, plan->thread_block>>>(d_y, mat->d_sellcs_val, mat->d_sellcs_col, mat->d_cl, mat->d_cs, d_x, mat->d_s_write_permutation, mat->d_s_write_permutation_offset, mat->c_size * mat->chunk, mat->seg_size);
    }
    else if (mat->block_size == 11) {
        kernel_spmv_amb_atomic11<<<plan->thread_grid, plan->thread_block>>>(d_y, mat->d_sellcs_val, mat->d_sellcs_col, mat->d_cl, mat->d_cs, d_x, mat->d_s_write_permutation, mat->d_s_write_permutation_offset, mat->c_size * mat->chunk, mat->seg_size);
    }
    else if (mat->block_size == 12) {
        kernel_spmv_amb_atomic12<<<plan->thread_grid, plan->thread_block>>>(d_y, mat->d_sellcs_val, mat->d_sellcs_col, mat->d_cl, mat->d_cs, d_x, mat->d_s_write_permutation, mat->d_s_write_permutation_offset, mat->c_size * mat->chunk, mat->seg_size);
    }
    else if (mat->block_size == 13) {
        kernel_spmv_amb_atomic13<<<plan->thread_grid, plan->thread_block>>>(d_y, mat->d_sellcs_val, mat->d_sellcs_col, mat->d_cl, mat->d_cs, d_x, mat->d_s_write_permutation, mat->d_s_write_permutation_offset, mat->c_size * mat->chunk, mat->seg_size);
    }
    else if (mat->block_size == 14) {
        kernel_spmv_amb_atomic14<<<plan->thread_grid, plan->thread_block>>>(d_y, mat->d_sellcs_val, mat->d_sellcs_col, mat->d_cl, mat->d_cs, d_x, mat->d_s_write_permutation, mat->d_s_write_permutation_offset, mat->c_size * mat->chunk, mat->seg_size);
    }
    else if (mat->block_size == 15) {
        kernel_spmv_amb_atomic15<<<plan->thread_grid, plan->thread_block>>>(d_y, mat->d_sellcs_val, mat->d_sellcs_col, mat->d_cl, mat->d_cs, d_x, mat->d_s_write_permutation, mat->d_s_write_permutation_offset, mat->c_size * mat->chunk, mat->seg_size);
    }
    else if (mat->block_size == 16) {
        kernel_spmv_amb_atomic16<<<plan->thread_grid, plan->thread_block>>>(d_y, mat->d_sellcs_val, mat->d_sellcs_col, mat->d_cl, mat->d_cs, d_x, mat->d_s_write_permutation, mat->d_s_write_permutation_offset, mat->c_size * mat->chunk, mat->seg_size);
    }
    else if (mat->block_size == 17) {
        kernel_spmv_amb_atomic17<<<plan->thread_grid, plan->thread_block>>>(d_y, mat->d_sellcs_val, mat->d_sellcs_col, mat->d_cl, mat->d_cs, d_x, mat->d_s_write_permutation, mat->d_s_write_permutation_offset, mat->c_size * mat->chunk, mat->seg_size);
    }
    else if (mat->block_size == 18) {
        kernel_spmv_amb_atomic18<<<plan->thread_grid, plan->thread_block>>>(d_y, mat->d_sellcs_val, mat->d_sellcs_col, mat->d_cl, mat->d_cs, d_x, mat->d_s_write_permutation, mat->d_s_write_permutation_offset, mat->c_size * mat->chunk, mat->seg_size);
    }
    else if (mat->block_size == 19) {
        kernel_spmv_amb_atomic19<<<plan->thread_grid, plan->thread_block>>>(d_y, mat->d_sellcs_val, mat->d_sellcs_col, mat->d_cl, mat->d_cs, d_x, mat->d_s_write_permutation, mat->d_s_write_permutation_offset, mat->c_size * mat->chunk, mat->seg_size);
    }
    else if (mat->block_size == 20) {
        kernel_spmv_amb_atomic20<<<plan->thread_grid, plan->thread_block>>>(d_y, mat->d_sellcs_val, mat->d_sellcs_col, mat->d_cl, mat->d_cs, d_x, mat->d_s_write_permutation, mat->d_s_write_permutation_offset, mat->c_size * mat->chunk, mat->seg_size);
    }
    if (cudaSuccess != cudaGetLastError()) {printf("Kernel error\n"); exit(0);}
    cudaThreadSynchronize();

}

