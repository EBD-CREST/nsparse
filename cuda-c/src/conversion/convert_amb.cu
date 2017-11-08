#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>

#include <math.h>

#include <cuda.h>
#include <helper_cuda.h>

#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

#include <nsparse.h>

#define AT

int memory_access;

__global__ void zero_fill_int(int *d_array, int size) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= size) {
        return;
    }

    d_array[i] = 0;
  
}

__global__ void set_permutation(int *d_permutation, int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= M) {
        return;
    }
  
    d_permutation[i] = i;
}

/* For SELL-C-sigma and S-SELL-C-sigma formats */
__global__ void set_cl(int *nnz_num, int *cl, int chunk, int pad_M)
{
    int c_size = pad_M / chunk;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= c_size) {
        return;
    }
    int offset = chunk * i;
    int max = 0;
    int j, length;
    for (j = 0; j < chunk; j++) {
        length = nnz_num[offset + j];
        if (length > max) {
            max = length;
        }
    }
    cl[i] = max;
}

__global__ void init_cs(int *d_cl, int *d_cs, int c_size, int chunk)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= c_size) {
        return;
    }

    if (i == 0) {
        d_cs[i] = 0;
    }
    else {
        d_cs[i] = d_cl[i - 1] * chunk;
    }

}

void set_sellcs_chunk(int *d_nnz_num, int *d_cl, int *d_cs, int *elements_num,
                      int total_pad_row_num, int chunk)
{
    size_t GS, BS;
    int r_size, c_size;
  
    c_size = total_pad_row_num / chunk;
  
    BS = MAX_LOCAL_THREAD_NUM;
    GS = div_round_up(c_size, BS);

    set_cl<<<GS, BS>>>(d_nnz_num, d_cl, chunk, total_pad_row_num);
    init_cs<<<GS, BS>>>(d_cl, d_cs, c_size, chunk);
    thrust::inclusive_scan(thrust::device, d_cs, d_cs + c_size, d_cs);

    /*Get elements_num*/
    checkCudaErrors(cudaMemcpy(elements_num, d_cs + (c_size - 1), sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&r_size, d_cl + (c_size - 1), sizeof(int), cudaMemcpyDeviceToHost));

    *elements_num += r_size * chunk;
}

__global__ void set_sellcs_col_val(int *d_rpt, int *d_col, real *d_val,
                                   int *d_nnz_num, int *d_write_permutation,
                                   int *d_sellcs_col, real *d_sellcs_val,
                                   int *d_cl, int *d_cs,
                                   int group_num_col, int pad_M, int M, int chunk)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= pad_M * group_num_col) {
        return;
    }

    int bid = i / chunk;
    int tid = i % chunk;
  
    int j;
    int width = d_cl[bid];
    int nnz_width = d_nnz_num[i];

    for(j=0; j<width; j++) {
        if(j < nnz_width) {
            d_sellcs_val[d_cs[bid] + tid + j * chunk] =
                d_val[d_rpt[d_write_permutation[i]] + j];
            d_sellcs_col[d_cs[bid] + tid + j * chunk] =
                d_col[d_rpt[d_write_permutation[i]] + j];

        }
        else {
            d_sellcs_val[d_cs[bid] + tid + j * chunk] = 0;
            d_sellcs_col[d_cs[bid] + tid + j * chunk] = 
                d_col[d_rpt[d_write_permutation[bid * chunk]] + j];
        }
    }
}

__global__ void set_segmented_nnz_num(int *d_rpt, int *d_col, int *d_nnz_num,
                                      int *d_group_seg, int *d_offset,
                                      size_t seg_size, size_t seg_num,
                                      int M, int pad_M, int group_num_col)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
  
    if (i >= M) {
        return;
    }

    int width = d_rpt[i + 1] - d_rpt[i];

    int g, j;
    int col;

    int offset = d_rpt[i];
    int index;

    for (j = 0; j < width; j++) {
        index = offset + j;
        col = d_col[index];
        g = col / seg_size;
        d_offset[index] = d_nnz_num[g * pad_M + i];
        d_nnz_num[g * pad_M + i]++;
        d_group_seg[index] = g;
    }
}

__global__ void init_segmented_rpt(int *d_nnz_num, int *d_seg_rpt, int total_pad_row_num)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > total_pad_row_num) {
        return;
    }

    if (i == 0) {
        d_seg_rpt[i] = 0;
    }

    else {
        d_seg_rpt[i] = d_nnz_num[i - 1];
    }
}

__global__ void set_segmented_col_val(int *d_rpt, int *d_col, real *d_val,
                                      int *d_seg_rpt, int *d_seg_col, real *d_seg_val,
                                      int *d_group_seg, int *d_offset,
                                      int M, int pad_M)
{
    int i = blockIdx.x;

    if(i >= M) {
        return;
    }
  
    int width = d_rpt[i + 1] - d_rpt[i];
  
    int j = threadIdx.x;
    int bs = blockDim.x;
    int index;
    for (; j < width; j += bs) {
        index = d_rpt[i] + j;
        d_seg_col[d_seg_rpt[d_group_seg[index] * pad_M + i] + d_offset[index]]
            = d_col[index];
        d_seg_val[d_seg_rpt[d_group_seg[index] * pad_M + i] + d_offset[index]]
            = d_val[index];
    }
}

void convert_segmented_csr(sfCSR *csr_mat,
                           int *d_nnz_num, int *d_seg_rpt, int *d_seg_col, real *d_seg_val,
                           size_t seg_size, size_t seg_num,
                           int M, int pad_M, int group_num_col)
{
    size_t GS, BS;
  
    int nz;
    int total_pad_row_num;
  
    int *d_group_seg, *d_offset;
  
    nz = csr_mat->nnz;
    total_pad_row_num = pad_M * group_num_col;
  
    checkCudaErrors(cudaMalloc((void **)&d_group_seg, sizeof(int) * nz));
    checkCudaErrors(cudaMalloc((void **)&d_offset, sizeof(int) * nz));
  
    BS = MAX_LOCAL_THREAD_NUM;
    GS = div_round_up(total_pad_row_num, BS);
    zero_fill_int<<<GS, BS>>>(d_nnz_num, total_pad_row_num);
  
    BS = WARP;
    GS = div_round_up(M, BS);
    set_segmented_nnz_num<<<GS, BS>>>(csr_mat->d_rpt, csr_mat->d_col, d_nnz_num, d_group_seg, d_offset, seg_size, seg_num, M, pad_M, group_num_col);
  
    /*Set segmented rpt*/
    GS = div_round_up((pad_M * group_num_col + 1), BS);

    init_segmented_rpt<<<GS, BS>>>(d_nnz_num, d_seg_rpt, pad_M * group_num_col);
    cudaThreadSynchronize();

    thrust::inclusive_scan(thrust::device, d_seg_rpt, d_seg_rpt + pad_M * group_num_col + 1, d_seg_rpt);
  
    /*Set segmented col and val*/
    GS = M;
    set_segmented_col_val<<<GS, BS>>>(csr_mat->d_rpt, csr_mat->d_col, csr_mat->d_val, d_seg_rpt, d_seg_col, d_seg_val, d_group_seg, d_offset, M, pad_M);

    cudaThreadSynchronize();

    checkCudaErrors(cudaFree(d_offset));
    checkCudaErrors(cudaFree(d_group_seg));

}

__global__ void update_write_permutation(int *write_permutation, int *nnz_num,
                                         int total_pad_row_num, int pad_M)
{  
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= total_pad_row_num) {
        return;
    }

    write_permutation[i] -= (i / pad_M) * pad_M;
}

__global__ void compress_write_permutation(int *d_write_permutation,
                                           int *d_full_write_permutation,
                                           int *d_gcs,
                                           int total_pad_row_num, int chunk)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_pad_row_num) {
        return;
    }

    int chunk_id = i / chunk;
    if (d_gcs[chunk_id + 1] - d_gcs[chunk_id] > 0) {
        int tid = i % chunk;
        d_write_permutation[d_gcs[chunk_id] * chunk + tid] = d_full_write_permutation[i];
    }
}

__global__ void compress_s_write_permutation(unsigned short *d_s_write_permutation,
                                             unsigned short *d_s_write_permutation_offset,
                                             int *d_write_permutation,
                                             int c_size, int chunk)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= c_size * chunk) {
        return;
    }

    int chunk_id = i / chunk;
    d_s_write_permutation[i] = (unsigned short)(d_write_permutation[i] % USHORT_MAX);
    if (i % chunk == 0) {
        d_s_write_permutation_offset[chunk_id] = (unsigned short)(d_write_permutation[i] / USHORT_MAX);
    }
}


/* For Segmented SELL-C-sigma format */
__global__ void get_c_size(int *d_c_size, int *d_full_cl, int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= size) {
        return;
    }

    if (d_full_cl[i] != 0) {
        atomicAdd(d_c_size, 1);
    }
}

__global__ void set_ushort_col(unsigned short *d_us_sellcs_col, int *d_sellcs_col,
                               int *d_cs, int *d_cl, BOOL *d_is_empty,
                               int group_num_col, int pad_M, int chunk, int seg_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= group_num_col * pad_M) {
        return;
    }

    int offset = i % chunk;
    int chunk_id = i / chunk;
    int width = d_cl[chunk_id];

    int j;
    int adr;
    for (j = 0; j < width; j++) {
        adr = d_cs[chunk_id] + j * chunk + offset;
        d_us_sellcs_col[adr] = (unsigned short)(d_sellcs_col[adr] % seg_size);
    }

    if (offset == 0) {
        int c_offset = 0;
        if (width > 0) {
            c_offset = (d_sellcs_col[d_cs[chunk_id]] / seg_size) << SCL_BORDER;
            d_is_empty[chunk_id] = FALSE;
            d_cl[chunk_id] = (width - 1) | c_offset;
        }
        else {
            d_is_empty[chunk_id] = TRUE;
            d_cl[chunk_id] = (width) | c_offset;
        }
    }
}

__global__ void init_gcs(int *d_gcs, BOOL *d_is_empty, int chunk_num)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= chunk_num) {
        return;
    }
  
    if (i == 0) d_gcs[i] = 0;

    if (d_is_empty[i] == TRUE) d_gcs[i + 1] = 0;
    else d_gcs[i + 1] = 1;
}


void set_gcs(int *d_gcs, BOOL *d_is_empty, int chunk_num)
{
    size_t GS, BS;
  
    BS = 256;
    GS = div_round_up(chunk_num, BS);
    init_gcs<<<GS, BS>>>(d_gcs, d_is_empty, chunk_num);
    thrust::inclusive_scan(thrust::device, d_gcs, d_gcs + chunk_num + 1, d_gcs);
}

__global__ void set_packed_cl_cs(int *d_packed_cl, int *d_packed_cs,
                                 int *d_cl, int *d_cs, int *d_gcs,
                                 int chunk_num)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= chunk_num) {
        return;
    }

    if (d_gcs[i + 1] - d_gcs[i] > 0) {
        d_packed_cl[d_gcs[i]] = d_cl[i];
        d_packed_cs[d_gcs[i]] = d_cs[i];
    }
}

__global__ void set_blocked_cl(unsigned int *d_blocked_cl,
                               int *d_packed_cl, int *d_packed_cs,
                               unsigned short *d_s_col_ell,
                               int c_size, int chunk, int block_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= c_size * chunk) {
        return;
    }

    int k;
    int cl_width = (d_packed_cl[i / chunk] & SCL_BIT) + 1;
    int max_width = 0;
    int j = i % chunk;
    int width = 0;
    int offset = d_packed_cs[i / chunk];
    int base = d_s_col_ell[offset + j];
    for (k = 1; k < cl_width; k++) {
        if (d_s_col_ell[offset + j + k * chunk] - base >= block_size) {
            base = d_s_col_ell[offset + j + k * chunk];
            width += block_size;
        }
    }
    width += block_size;
  
    int shfl_width;
    shfl_width = __shfl_xor(width, 16);
    width = (width < shfl_width)? shfl_width : width;
    shfl_width = __shfl_xor(width, 8);
    width = (width < shfl_width)? shfl_width : width;
    shfl_width = __shfl_xor(width, 4);
    width = (width < shfl_width)? shfl_width : width;
    shfl_width = __shfl_xor(width, 2);
    width = (width < shfl_width)? shfl_width : width;
    shfl_width = __shfl_xor(width, 1);
    width = (width < shfl_width)? shfl_width : width;

    if (j == 0) {
        max_width = width;
        d_blocked_cl[i / chunk] = ((max_width / block_size) - 1) | ((d_packed_cl[i / chunk] >> SCL_BORDER) << SCL_BORDER);
    }
}

__global__ void init_blocked_cs(int *d_cs, unsigned int *d_cl,
                                int c_size, int chunk, int block_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= c_size) {
        return;
    }

    if (i == 0) {
        d_cs[i] = 0;
    }
    else {
        d_cs[i] = ((d_cl[i - 1] & SCL_BIT) + 1) * chunk * block_size;
    }
}

void set_blocked_cl_cs(unsigned int *d_blocked_cl, int *d_blocked_cs,
                       int *d_packed_cl, int *d_packed_cs,
                       unsigned short *d_s_col_ell,
                       int c_size, int chunk,
                       int block_size, int *c_nnz)
{
    size_t GS, BS;
  
    BS = 256;
    GS = div_round_up((c_size * chunk), BS);

    set_blocked_cl<<<GS, BS>>>(d_blocked_cl, d_packed_cl, d_packed_cs, d_s_col_ell, c_size, chunk, block_size);

    GS = div_round_up(c_size, BS);
    init_blocked_cs<<<GS, BS>>>(d_blocked_cs, d_blocked_cl, c_size, chunk, block_size);

    thrust::inclusive_scan(thrust::device, d_blocked_cs, d_blocked_cs + c_size, d_blocked_cs);

    int r_size;
    checkCudaErrors(cudaMemcpy(c_nnz, d_blocked_cs + (c_size - 1), sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&r_size, d_blocked_cl + (c_size - 1), sizeof(unsigned int), cudaMemcpyDeviceToHost));

    *c_nnz += ((r_size & SCL_BIT) + 1) * block_size * chunk;

}

__global__ void set_blocked_col_val(unsigned short *d_bs_col_ell, real *d_b_val_ell,
                                    unsigned int *d_blocked_cl, int *d_blocked_cs,
                                    int *d_packed_cl, int *d_packed_cs,
                                    unsigned short *d_s_col_ell, real *d_val_ell,
                                    int c_size, int chunk, int block_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= c_size * chunk) {
        return;
    }

    int chunk_id = i / chunk;
    int tid = i % chunk;
    int cl_width = ((d_blocked_cl[chunk_id] & SCL_BIT) + 1) * block_size;

    int base;
    int h, k;
    int it = 0;
    int c;
    for (k = 0; k < cl_width / block_size; k++) {
        if (it < ((d_packed_cl[chunk_id] & SCL_BIT) + 1)) {
            c = d_s_col_ell[d_packed_cs[chunk_id] + tid + it * chunk];
            base = c;
            d_bs_col_ell[d_blocked_cs[chunk_id] / block_size + tid + k * chunk] = base;
            for (h = 0; h < c - base; h++) {
                d_b_val_ell[d_blocked_cs[chunk_id] + tid + (k * block_size + h) * chunk] = 0;
            }
            d_b_val_ell[d_blocked_cs[chunk_id] + tid + (k * block_size + (c - base)) * chunk] = d_val_ell[d_packed_cs[chunk_id] + tid + it * chunk];
            it++;
            for (h = c - base + 1; h < block_size; h++) {
                if (it < ((d_packed_cl[chunk_id] & SCL_BIT) + 1)) {
                    if (d_s_col_ell[d_packed_cs[chunk_id] + tid + it * chunk] - base == h) {
                        d_b_val_ell[d_blocked_cs[chunk_id] + tid + (k * block_size + h) * chunk] = d_val_ell[d_packed_cs[chunk_id] + tid + it * chunk];
                        it++;
                    }
                    else {
                        d_b_val_ell[d_blocked_cs[chunk_id] + tid + (k * block_size + h) * chunk] = 0;
                    }
                }
                else {
                    d_b_val_ell[d_blocked_cs[chunk_id] + tid + (k * block_size + h) * chunk] = 0;
                }
            }
        }
        else {
            d_bs_col_ell[d_blocked_cs[chunk_id] / block_size + tid + k * chunk] = (d_s_col_ell[d_packed_cs[chunk_id] + tid + ((d_packed_cl[chunk_id] & SCL_BIT)) * chunk] / block_size) * block_size;
            for (h = 0; h < block_size; h++) {
                d_b_val_ell[d_blocked_cs[chunk_id] + tid + (k * block_size + h) * chunk] = 0;
            }
        }
    }
}

__global__ void set_d_check_nnz(int *d_check_nnz, int *d_nnz_num,
                                int pad_M, int SIGMA, int sigma_block_row)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pad_M) {
        return;
    }

    int a = 1;
    if (d_nnz_num[blockIdx.y * pad_M + i] > 0) {
        atomicAdd(&(d_check_nnz[blockIdx.y * sigma_block_row + i / SIGMA]), a);
    }
}


void set_check_nnz(int *d_check_nnz, int *d_nnz_num,
                   int sigma_block, int pad_M, int SIGMA, int group_num_col)
{
    int GS, BS;
    BS = MAX_LOCAL_THREAD_NUM;
    GS = div_round_up(sigma_block, BS);
  
    zero_fill_int<<<GS, BS>>>(d_check_nnz, sigma_block);

    GS = div_round_up(pad_M, BS);
    set_d_check_nnz<<<dim3(GS, group_num_col), dim3(BS, 1)>>>(d_check_nnz, d_nnz_num, pad_M, SIGMA, div_round_up(pad_M, SIGMA));
  
}

void evaluate_spmv(sfAMB *mat, real *d_x, real *d_y,
                   float *min_msec, sfPlan *plan)
{
    int i, coe;
    float msec, ave_msec, best_msec;
    sfPlan plan_;
    best_msec = sfFLT_MAX;
    cudaEvent_t event[2];
    for (i = 0; i < 2; i++) {
        cudaEventCreate(&(event[i]));
    }

    for (coe = 2; coe <= MAX_THREAD_BLOCK; coe *= 2) {
        plan_.thread_block = WARP * coe;
        plan_.thread_grid = div_round_up(mat->chunk * mat->c_size, plan_.thread_block);

        ave_msec = 0;
        for (i = 0; i < TEST_NUM; i++) {
            cudaEventRecord(event[0], 0);
            sf_spmv_amb(d_y, mat, d_x, &plan_);
            cudaEventRecord(event[1], 0);
            cudaThreadSynchronize();
	
            cudaEventElapsedTime(&msec, event[0], event[1]);
	
            if (i > 0) {
                ave_msec += msec;
            }
        }
        ave_msec /= TEST_NUM - 1;
        if (*min_msec > ave_msec) {
            *min_msec = ave_msec;
            plan->seg_size = (mat->seg_size);
            plan->block_size = mat->block_size;
            plan->thread_block = plan_.thread_block;
            plan->thread_grid = plan_.thread_grid;
        }
        if (best_msec > ave_msec) {
            best_msec = ave_msec;
        }
    }
    cudaEventDestroy(event[0]);
    cudaEventDestroy(event[1]);
  
}



void convert_amb_at(sfCSR *csr_mat,
                    sfAMB *mat,
                    real *d_x, real *d_y,
                    float *min_msec,
                    sfPlan *plan)
{
    int GS, BS;
    int i;
  
    int M, pad_M, nz, nnz, c_nnz;
    int total_pad_row_num;

    int SIGMA, chunk;
    int start, end;
    int block_size;
    int sigma_block;

    int *d_nnz_num;
    int *d_seg_rpt;
    int *d_seg_col;
    real *d_seg_val;
    int *d_col_ell;
    int *d_full_write_permutation;
    int *d_full_cs, *d_full_cl;
    BOOL *d_is_empty;
    int *d_packed_cl, *d_packed_cs;
    unsigned short *d_bs_col_ell;
    real *d_b_val_ell;

    int *check_nnz;
    int *d_check_nnz;
    unsigned short *d_nbs_sellcs_col;
    real *d_nb_sellcs_val;
  
    int *d_c_size;
    int *d_gcs;

    M = mat->M;
    nz = csr_mat->nnz;
    pad_M = mat->pad_M;
    chunk = mat->chunk;
    SIGMA = mat->SIGMA;
    total_pad_row_num = pad_M * mat->group_num_col;
    BS = MAX_LOCAL_THREAD_NUM;

    /* Step 1 : Convert the matrix into Segmented Format */
    /*Convert format from CSR to Segmented CSR*/
    checkCudaErrors(cudaMalloc((void **)&d_nnz_num, sizeof(int) * total_pad_row_num));
    checkCudaErrors(cudaMalloc((void **)&d_seg_rpt, sizeof(int) * (pad_M * mat->group_num_col + 1)));
    checkCudaErrors(cudaMalloc((void **)&d_seg_col, sizeof(int) * nz));
    checkCudaErrors(cudaMalloc((void **)&d_seg_val, sizeof(real) * nz));
  
    convert_segmented_csr(csr_mat, d_nnz_num, d_seg_rpt, d_seg_col, d_seg_val, mat->seg_size, mat->seg_num, M, pad_M, mat->group_num_col);
  
    /*Set permutation*/
    checkCudaErrors(cudaMalloc((void **)&(d_full_write_permutation), sizeof(int) * total_pad_row_num));

    GS = div_round_up(total_pad_row_num, BS);
    set_permutation<<<GS, BS>>>(d_full_write_permutation, total_pad_row_num);
  
    sigma_block = div_round_up(pad_M, SIGMA) * mat->group_num_col;
    check_nnz = (int *)malloc(sizeof(int) * sigma_block);
    checkCudaErrors(cudaMalloc((void **)&(d_check_nnz), sizeof(int) * sigma_block));
    set_check_nnz(d_check_nnz, d_nnz_num, sigma_block, pad_M, SIGMA, mat->group_num_col);
    checkCudaErrors(cudaMemcpy(check_nnz, d_check_nnz, sizeof(int) * sigma_block, cudaMemcpyDeviceToHost));
  
    /*Sorting each sigma rows*/
    if (M < SIGMA) {
        SIGMA = M;
    }
    /*Sorting each M rows*/
    if (SIGMA > 1) {
        thrust::device_ptr<int> dev_nnz_num(d_nnz_num);
        thrust::device_ptr<int> dev_full_write_permutation(d_full_write_permutation);
        start = 0;
        for (i = 0; i < mat->group_num_col; i++) {
            start = 0;
            end = 0;
            while (start < M) {
                end += SIGMA;
                if (end >= M) {
                    end = M;
                }
                if (check_nnz[i * div_round_up(pad_M, SIGMA) + start / SIGMA] > 0) {
                    thrust::stable_sort_by_key(dev_nnz_num + i * pad_M + start,
                                               dev_nnz_num + i * pad_M + end,
                                               dev_full_write_permutation + i * pad_M + start,
                                               thrust::greater<int>());
                }
                start += SIGMA;
            }
        }
    }

    /*Set chunk size*/
    checkCudaErrors(cudaMalloc((void **)&d_full_cl, sizeof(int) * total_pad_row_num / chunk));
    checkCudaErrors(cudaMalloc((void **)&d_full_cs, sizeof(int) * total_pad_row_num / chunk));
  
    set_sellcs_chunk(d_nnz_num, d_full_cl, d_full_cs, &nnz, total_pad_row_num, chunk);

    /*Set sellcs_col and sellcs_val*/
    checkCudaErrors(cudaMalloc((void **)&(d_col_ell), sizeof(int) * nnz));
    checkCudaErrors(cudaMalloc((void **)&(d_nb_sellcs_val), sizeof(real) * nnz));
  
    GS = div_round_up(total_pad_row_num, BS);
    set_sellcs_col_val<<<GS, BS>>>(d_seg_rpt, d_seg_col, d_seg_val, d_nnz_num, d_full_write_permutation, d_col_ell, d_nb_sellcs_val, d_full_cl, d_full_cs, mat->group_num_col, pad_M, M, chunk);

    cudaFree(d_seg_rpt);
    cudaFree(d_seg_col);
    cudaFree(d_seg_val);

    /* Compression */
    mat->c_size = 0;
    checkCudaErrors(cudaMalloc((void **)&d_c_size, sizeof(int)));
    checkCudaErrors(cudaMemcpy(d_c_size, &(mat->c_size), sizeof(int), cudaMemcpyHostToDevice));
    get_c_size<<<div_round_up((total_pad_row_num / chunk), BS), BS>>>
        (d_c_size, d_full_cl, total_pad_row_num / chunk);
    checkCudaErrors(cudaMemcpy(&(mat->c_size), d_c_size, sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree(d_c_size);
  
    checkCudaErrors(cudaMalloc((void **)&(d_nbs_sellcs_col), sizeof(unsigned short) * nnz));
  
    checkCudaErrors(cudaMalloc((void **)&(d_is_empty), sizeof(BOOL) * mat->group_num_col * pad_M / chunk));
  
    set_ushort_col<<<div_round_up(total_pad_row_num, BS), BS>>>(d_nbs_sellcs_col, d_col_ell, d_full_cs, d_full_cl, d_is_empty, mat->group_num_col, pad_M, chunk, mat->seg_size);
  
    checkCudaErrors(cudaMalloc((void **)&d_gcs, sizeof(int) * (total_pad_row_num / chunk + 1)));
  
    set_gcs(d_gcs, d_is_empty, total_pad_row_num / chunk);

    checkCudaErrors(cudaMalloc((void **)&d_packed_cl, sizeof(int) * mat->c_size));
    checkCudaErrors(cudaMalloc((void **)&d_packed_cs, sizeof(int) * mat->c_size));

    set_packed_cl_cs<<<div_round_up((total_pad_row_num / chunk), BS), BS>>>(d_packed_cl, d_packed_cs, d_full_cl, d_full_cs, d_gcs, total_pad_row_num / chunk);

    cudaFree(d_full_cl);
    cudaFree(d_full_cs);
    cudaFree(d_col_ell);
  
    /* Updating the write permutation */
    update_write_permutation<<<div_round_up(total_pad_row_num, BS), BS>>>(d_full_write_permutation, d_nnz_num, total_pad_row_num, pad_M);
    checkCudaErrors(cudaMalloc((void **)&(mat->d_write_permutation), sizeof(int) * (mat->c_size * chunk)));
    compress_write_permutation<<<div_round_up(total_pad_row_num, BS), BS>>>(mat->d_write_permutation, d_full_write_permutation, d_gcs, total_pad_row_num, chunk);

    checkCudaErrors(cudaMalloc((void **)&(mat->d_s_write_permutation), sizeof(unsigned short) * (mat->c_size * chunk)));
    checkCudaErrors(cudaMalloc((void **)&(mat->d_s_write_permutation_offset), sizeof(unsigned short) * mat->c_size));
    compress_s_write_permutation<<<div_round_up((mat->c_size * chunk), BS), BS>>>(mat->d_s_write_permutation, mat->d_s_write_permutation_offset, mat->d_write_permutation, mat->c_size, mat->chunk);
  
    cudaFree(d_nnz_num);
    cudaFree(d_full_write_permutation);
    cudaFree(d_gcs);

    /* Blocking */
    if (plan->isPlan == FALSE) {
        for (block_size = 1; block_size <= MAX_BLOCK_SIZE; block_size++) {
    
            mat->block_size = block_size;

            if (block_size > 1) {
                checkCudaErrors(cudaFree(mat->d_cl));
                checkCudaErrors(cudaFree(mat->d_cs));
                checkCudaErrors(cudaFree(mat->d_sellcs_col));
                checkCudaErrors(cudaFree(mat->d_sellcs_val));
            }
            checkCudaErrors(cudaMalloc((void **)&(mat->d_cl), sizeof(unsigned int) * mat->c_size));
            checkCudaErrors(cudaMalloc((void **)&(mat->d_cs), sizeof(int) * mat->c_size));
    
            c_nnz = 0;
            set_blocked_cl_cs(mat->d_cl, mat->d_cs, d_packed_cl, d_packed_cs, d_nbs_sellcs_col, mat->c_size, chunk, block_size, &c_nnz);

            mat->nnz = c_nnz;
            checkCudaErrors(cudaMalloc((void **)&(mat->d_sellcs_col), sizeof(unsigned short) * c_nnz / block_size));
            checkCudaErrors(cudaMalloc((void **)&(mat->d_sellcs_val), sizeof(real) * c_nnz));
    
            set_blocked_col_val<<<div_round_up((mat->c_size * chunk), BS), BS>>>(mat->d_sellcs_col, mat->d_sellcs_val, mat->d_cl, mat->d_cs, d_packed_cl, d_packed_cs, d_nbs_sellcs_col, d_nb_sellcs_val, mat->c_size, chunk, block_size);
            cudaThreadSynchronize();
    
#ifdef AT
            evaluate_spmv(mat, d_x, d_y, min_msec, plan);
#else 
            int footprint = 0;
            footprint += (mat->nnz / mat->block_size) * sizeof(unsigned short); // col
            footprint += (mat->nnz) * sizeof(real); // val
            footprint += (mat->c_size) * sizeof(int) * 2; // cs, cl
            footprint += (mat->c_size) * mat->chunk * sizeof(unsigned short) + (mat->c_size) * sizeof(unsigned short); // permutation
            footprint += (mat->c_size) * mat->chunk * sizeof(real) * 2; // output
            footprint += (mat->M) * sizeof(real) * 2; // input + output_init
            // printf("%s, %d, %d, %d\n", mat->matrix_name, mat->seg_size, mat->block_size, footprint);
            if (memory_access > footprint) {
                memory_access = footprint;
                plan->seg_size = (mat->seg_size);
                plan->block_size = mat->block_size;
            }

#endif
        }

        /* Set Best Block Size */
        mat->block_size = plan->block_size;
        checkCudaErrors(cudaFree(mat->d_cl));
        checkCudaErrors(cudaFree(mat->d_cs));
        checkCudaErrors(cudaFree(mat->d_sellcs_col));
        checkCudaErrors(cudaFree(mat->d_sellcs_val));
    }

    checkCudaErrors(cudaMalloc((void **)&(mat->d_cl), sizeof(unsigned int) * mat->c_size));
    checkCudaErrors(cudaMalloc((void **)&(mat->d_cs), sizeof(int) * mat->c_size));

    c_nnz = 0;
    set_blocked_cl_cs(mat->d_cl, mat->d_cs, d_packed_cl, d_packed_cs, d_nbs_sellcs_col, mat->c_size, chunk, mat->block_size, &c_nnz);

    checkCudaErrors(cudaMalloc((void **)&d_bs_col_ell, sizeof(unsigned short) * (c_nnz / (mat->block_size))));
    checkCudaErrors(cudaMalloc((void **)&d_b_val_ell, sizeof(real) * c_nnz));
  
    mat->nnz = c_nnz;
    checkCudaErrors(cudaMalloc((void **)&(mat->d_sellcs_col), sizeof(unsigned short) * c_nnz / (mat->block_size)));
    checkCudaErrors(cudaMalloc((void **)&(mat->d_sellcs_val), sizeof(real) * c_nnz));

    set_blocked_col_val<<<div_round_up((mat->c_size * chunk), BS), BS>>>(mat->d_sellcs_col, mat->d_sellcs_val, mat->d_cl, mat->d_cs, d_packed_cl, d_packed_cs, d_nbs_sellcs_col, d_nb_sellcs_val, mat->c_size, chunk, (mat->block_size));
    
    checkCudaErrors(cudaFree(d_b_val_ell));
    checkCudaErrors(cudaFree(d_bs_col_ell));

    cudaFree(d_packed_cl);
    cudaFree(d_packed_cs);
    cudaFree(d_nb_sellcs_val);
    cudaFree(d_nbs_sellcs_col);

}

void sf_csr2amb(sfAMB *mat,
                sfCSR *csr_mat,
                real *d_x,
                sfPlan *plan)
{
    int i;

    int max_div_size, seg_pattern;
    size_t *base_size;
    int seg_it;
    
    float min_msec;
    
    real *d_y;
    cudaEvent_t event[2];
    for (i = 0; i < 2; i++) {
        cudaEventCreate(&(event[i]));
    }
  
    memory_access = INT_MAX;
    min_msec = sfFLT_MAX;
  
    mat->M = csr_mat->M;
    mat->N = csr_mat->N;
    mat->chunk = WARP;
    mat->pad_M = mat->chunk * div_round_up(csr_mat->M, mat->chunk);
    mat->nnz_max = csr_mat->nnz_max;
    mat->matrix_name = csr_mat->matrix_name;
    mat->SIGMA = SHORT_MAX;
  
    checkCudaErrors(cudaMalloc((void **)&d_y, sizeof(real) * ((csr_mat->M) + WARP)));
  
    if (plan->isPlan == TRUE) {
        mat->seg_size = plan->seg_size;
        mat->seg_num = div_round_up(csr_mat->N, mat->seg_size);
        plan->seg_num = mat->seg_num;
        mat->group_num_col = mat->seg_num;
        mat->block_size = plan->block_size;
    
        // convert_amb_opt(csr_mat, mat);
        convert_amb_at(csr_mat, mat, d_x, d_y, &min_msec, plan);
        evaluate_spmv(mat, d_x, d_y, &min_msec, plan);
    }
    else {
        max_div_size = 128 * 1024;
        seg_pattern = (csr_mat->N < max_div_size)? 5 : 1;
        base_size = (size_t *)malloc(sizeof(size_t) * seg_pattern);
        base_size[0] = 64 * 1024;
        if (csr_mat->N < max_div_size) {
            for (i = 1; i < seg_pattern; i++) {
                base_size[i] = i * 1024;
            }
        }
        if (csr_mat->N < 100) {
            for (i = 1; i < seg_pattern; i++) {
                base_size[i] = i;
            }
        }

        for (seg_it = 0; seg_it < seg_pattern; seg_it++) {
            if (seg_it > 0) {
                release_amb(*mat);
            }
            (mat->seg_size) = base_size[seg_it];
            (mat->seg_num) = div_round_up(csr_mat->N, mat->seg_size);
      
            mat->group_num_col = (mat->seg_num);
  
            convert_amb_at(csr_mat, mat, d_x, d_y, &min_msec, plan);
        }
      
        plan->isPlan = TRUE;
        /* Set Best AMB format */
        if ((mat->seg_size != plan->seg_size)) {
            release_amb(*mat);
            (mat->seg_size) = plan->seg_size;
            (mat->seg_num) = div_round_up(csr_mat->N, (mat->seg_size));
            mat->block_size = plan->block_size;
            mat->group_num_col = (mat->seg_num);
      
            convert_amb_at(csr_mat, mat, d_x, d_y, &min_msec, plan);
        }

#ifdef AT
#else
        evaluate_spmv(mat, d_x, d_y, &min_msec, plan);
#endif

        plan->seg_num = (mat->seg_num);
  
    }

    cudaFree(d_y);

}
