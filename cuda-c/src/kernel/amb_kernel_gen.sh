out="./kernel_spmv_amb.cu"
max_block=20
block=1

echo "#include <stdio.h>
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

" > ${out}

while [ $block -le $max_block ]
do
    echo "__global__ void kernel_spmv_amb_atomic${block}(real *ans,
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
  int colstart = (cs[c_index] / $block) + (threadIdx.x & (WARP - 1));
  int length = cl[c_index];
  int width = length & SCL_BIT;
  int c_offset = (length >> SCL_BORDER) * seg_size;
  
  int h;
  int c = ld_gbl_ushort(col + colstart) + c_offset;
  real answer = ld_gbl_val(value + start) * vector[c];
  start += WARP;" >> ${out}

    it=1
    while [ $it -lt $block ]
    do
	echo "  answer += ld_gbl_val(value + start) * vector[c + ${it}];
  start += WARP;" >> ${out}
	it=`expr $it + 1`
    done
    echo "  colstart += WARP;

  for (h = 0; h < width; h++) {
    c = ld_gbl_ushort(col + colstart) + c_offset;
    answer += ld_gbl_val(value + start) * vector[c];
    start += WARP;" >> ${out}
    it=1
    while [ $it -lt $block ]
    do
	echo "    answer += ld_gbl_val(value + start) * vector[c + ${it}];
    start += WARP;" >> ${out}
	it=`expr $it + 1`
    done
    echo "    colstart += WARP;
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
" >> ${out}
    block=`expr $block + 1`
done

echo "void sf_spmv_amb(real *d_y, sfAMB *mat, real *d_x, sfPlan *plan) {
  
  kernel_spmv_init_ans<<<div_round_up(mat->M, MAX_LOCAL_THREAD_NUM), MAX_LOCAL_THREAD_NUM>>>(d_y, mat->M);

  if (mat->block_size == 1) {
    kernel_spmv_amb_atomic1<<<plan->thread_grid, plan->thread_block>>>(d_y, mat->d_sellcs_val, mat->d_sellcs_col, mat->d_cl, mat->d_cs, d_x, mat->d_s_write_permutation, mat->d_s_write_permutation_offset, mat->c_size * mat->chunk, mat->seg_size);
  }
" >> ${out}

block=2
while [ $block -le $max_block ]
do
    echo "  else if (mat->block_size == $block) {
    kernel_spmv_amb_atomic$block<<<plan->thread_grid, plan->thread_block>>>(d_y, mat->d_sellcs_val, mat->d_sellcs_col, mat->d_cl, mat->d_cs, d_x, mat->d_s_write_permutation, mat->d_s_write_permutation_offset, mat->c_size * mat->chunk, mat->seg_size);
  }" >> ${out}
    block=`expr $block + 1`
done

echo "  if (cudaSuccess != cudaGetLastError()) {printf(\"Kernel error\n\"); exit(0);}
  cudaThreadSynchronize();

}
" >> ${out}

