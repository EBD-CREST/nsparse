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

template <int block_size>
__global__ void kernel_spmv_amb_atomic(real *ans,
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
    int colstart = (cs[c_index] / block_size) + (threadIdx.x & (WARP - 1));

    int length = cl[c_index];
    int width = length & SCL_BIT;
    int c_offset = (length >> SCL_BORDER) * seg_size;
  
    int h;
    int c = ld_gbl_ushort(col + colstart) + c_offset;
    real answer = 0;
#pragma unroll
    for (int b = 0; b < block_size; ++b) {
        answer += ld_gbl_val(value + start) * vector[c + b];
        start += WARP;
    }
    colstart += WARP;

    for (h = 0; h < width; h++) {
        c = ld_gbl_ushort(col + colstart) + c_offset;
        for (int b = 0; b < block_size; ++b) {
            answer += ld_gbl_val(value + start) * vector[c + b];
            start += WARP;
        }
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

template <int id>
inline void call_kernel_spmv_amb_atomic(real *d_y, sfAMB *mat, real *d_x, sfPlan *plan)
{
    if (mat->block_size == id) {
        kernel_spmv_amb_atomic<id><<<plan->thread_grid, plan->thread_block>>>(d_y, mat->d_sellcs_val, mat->d_sellcs_col, mat->d_cl, mat->d_cs, d_x, mat->d_s_write_permutation, mat->d_s_write_permutation_offset, mat->c_size * mat->chunk, mat->seg_size);
    }
    call_kernel_spmv_amb_atomic<id + 1>(d_y, mat, d_x, plan);
}

template <>
inline void call_kernel_spmv_amb_atomic<MAX_BLOCK_SIZE>(real *d_y, sfAMB *mat, real *d_x, sfPlan *plan)
{
    if (mat->block_size == MAX_BLOCK_SIZE) {
        kernel_spmv_amb_atomic<MAX_BLOCK_SIZE><<<plan->thread_grid, plan->thread_block>>>(d_y, mat->d_sellcs_val, mat->d_sellcs_col, mat->d_cl, mat->d_cs, d_x, mat->d_s_write_permutation, mat->d_s_write_permutation_offset, mat->c_size * mat->chunk, mat->seg_size);
    }
}

void sf_spmv_amb(real *d_y, sfAMB *mat, real *d_x, sfPlan *plan) {
  
    kernel_spmv_init_ans<<<div_round_up(mat->M, MAX_LOCAL_THREAD_NUM), MAX_LOCAL_THREAD_NUM>>>(d_y, mat->M);
    call_kernel_spmv_amb_atomic<1>(d_y, mat, d_x, plan);
    cudaThreadSynchronize();

}

