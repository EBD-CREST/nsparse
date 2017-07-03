#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <cuda.h>
#include <helper_cuda.h>
#include <cusparse_v2.h>

#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include <nsparse.h>
#include <nsparse_asm.h>

/* SpGEMM Specific Parameters */
#define HASH_SCAL 107 // Set disjoint number to COMP_SH_SIZE
#define ONSTREAM

void init_bin(sfBIN *bin, int M)
{
    int i;
    bin->stream = (cudaStream_t *)malloc(sizeof(cudaStream_t) * BIN_NUM);
    for (i = 0; i < BIN_NUM; i++) {
        cudaStreamCreate(&(bin->stream[i]));
    }
  
    bin->bin_size = (int *)malloc(sizeof(int) * BIN_NUM);
    bin->bin_offset = (int *)malloc(sizeof(int) * BIN_NUM);
    checkCudaErrors(cudaMalloc((void **)&(bin->d_row_perm), sizeof(int) * M));
    checkCudaErrors(cudaMalloc((void **)&(bin->d_row_nz), sizeof(int) * (M + 1)));
    checkCudaErrors(cudaMalloc((void **)&(bin->d_max), sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&(bin->d_bin_size), sizeof(int) * BIN_NUM));
    checkCudaErrors(cudaMalloc((void **)&(bin->d_bin_offset), sizeof(int) * BIN_NUM));
    i = 0;
    bin->max_intprod = 0;
    bin->max_nz = 0;
}

void release_bin(sfBIN bin)
{
    int i;
    cudaFree(bin.d_row_nz);
    cudaFree(bin.d_row_perm);
    cudaFree(bin.d_max);
    cudaFree(bin.d_bin_size);
    cudaFree(bin.d_bin_offset);

    free(bin.bin_size);
    free(bin.bin_offset);
    for (i = 0; i < BIN_NUM; i++) {
        cudaStreamDestroy(bin.stream[i]);
    }
    free(bin.stream);
}

__global__ void set_intprod_num(int *d_arpt, int *d_acol,
                                const int* __restrict__ d_brpt,
                                int *d_row_intprod, int *d_max_intprod,
                                int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) {
        return;
    }
    int nz_per_row = 0;
    int j;
    for (j = d_arpt[i]; j < d_arpt[i + 1]; j++) {
        nz_per_row += d_brpt[d_acol[j] + 1] - d_brpt[d_acol[j]];
    }
    d_row_intprod[i] = nz_per_row;
    atomicMax(d_max_intprod, nz_per_row);
}

__global__ void set_bin(int *d_row_nz, int *d_bin_size, int *d_max,
                        int M, int min, int mmin)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) {
        return;
    }
    int nz_per_row = d_row_nz[i];

    atomicMax(d_max, nz_per_row);

    int j = 0;
    for (j = 0; j < BIN_NUM - 2; j++) {
        if (nz_per_row <= (min << j)) {
            if (nz_per_row <= (mmin)) {
                atomicAdd(d_bin_size + j, 1);
            }
            else {
                atomicAdd(d_bin_size + j + 1, 1);
            }
            return;
        }
    }
    atomicAdd(d_bin_size + BIN_NUM - 1, 1);
}

__global__ void init_row_perm(int *d_permutation, int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= M) {
        return;
    }
  
    d_permutation[i] = i;
}

__global__ void set_row_perm(int *d_bin_size, int *d_bin_offset,
                             int *d_max_row_nz, int *d_row_perm,
                             int M, int min, int mmin)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= M) {
        return;
    }

    int nz_per_row = d_max_row_nz[i];
    int dest;
  
    int j = 0;
    for (j = 0; j < BIN_NUM - 2; j++) {
        if (nz_per_row <= (min << j)) {
            if (nz_per_row <= mmin) {
                dest = atomicAdd(d_bin_size + j, 1);
                d_row_perm[d_bin_offset[j] + dest] = i;
            }
            else {
                dest = atomicAdd(d_bin_size + j + 1, 1);
                d_row_perm[d_bin_offset[j + 1] + dest] = i;
            }
            return;
        }
    }
    dest = atomicAdd(d_bin_size + BIN_NUM - 1, 1);
    d_row_perm[d_bin_offset[BIN_NUM - 1] + dest] = i;

}

void set_max_bin(int *d_arpt, int *d_acol, int *d_brpt, sfBIN *bin, int M)
{
    int i;
    int GS, BS;
  
    for (i = 0; i < BIN_NUM; i++) {
        bin->bin_size[i] = 0;
        bin->bin_offset[i] = 0;
    }
  
    cudaMemcpy(bin->d_bin_size, bin->bin_size, sizeof(int) * BIN_NUM, cudaMemcpyHostToDevice);
    cudaMemcpy(bin->d_max, &(bin->max_intprod), sizeof(int), cudaMemcpyHostToDevice);
  
    BS = 1024;
    GS = div_round_up(M, BS);
    set_intprod_num<<<GS, BS>>>(d_arpt, d_acol, d_brpt, bin->d_row_nz, bin->d_max, M);
    cudaMemcpy(&(bin->max_intprod), bin->d_max, sizeof(int), cudaMemcpyDeviceToHost);
  
    if (bin->max_intprod > IMB_PWMIN) {
        set_bin<<<GS, BS>>>(bin->d_row_nz, bin->d_bin_size, bin->d_max, M, IMB_MIN, IMB_PWMIN);
  
        cudaMemcpy(bin->bin_size, bin->d_bin_size, sizeof(int) * BIN_NUM, cudaMemcpyDeviceToHost);
        cudaMemcpy(bin->d_bin_size, bin->bin_offset, sizeof(int) * BIN_NUM, cudaMemcpyHostToDevice);

        for (i = 0; i < BIN_NUM - 1; i++) {
            bin->bin_offset[i + 1] = bin->bin_offset[i] + bin->bin_size[i];
        }
        cudaMemcpy(bin->d_bin_offset, bin->bin_offset, sizeof(int) * BIN_NUM, cudaMemcpyHostToDevice);

        set_row_perm<<<GS, BS>>>(bin->d_bin_size, bin->d_bin_offset, bin->d_row_nz, bin->d_row_perm, M, IMB_MIN, IMB_PWMIN);
    }
    else {
        bin->bin_size[0] = M;
        for (i = 1; i < BIN_NUM; i++) {
            bin->bin_size[i] = 0;
        }
        bin->bin_offset[0] = 0;
        for (i = 1; i < BIN_NUM; i++) {
            bin->bin_offset[i] = M;
        }
        init_row_perm<<<GS, BS>>>(bin->d_row_perm, M);
    }
}


void set_min_bin(sfBIN *bin, int M)
{
    int i;
    int GS, BS;
  
    for (i = 0; i < BIN_NUM; i++) {
        bin->bin_size[i] = 0;
        bin->bin_offset[i] = 0;
    }
  
    cudaMemcpy(bin->d_bin_size, bin->bin_size, sizeof(int) * BIN_NUM, cudaMemcpyHostToDevice);
    cudaMemcpy(bin->d_max, &(bin->max_nz), sizeof(int), cudaMemcpyHostToDevice);
  
    BS = 1024;
    GS = div_round_up(M, BS);
    set_bin<<<GS, BS>>>(bin->d_row_nz, bin->d_bin_size,
                        bin->d_max,
                        M, B_MIN, B_PWMIN);
  
    cudaMemcpy(&(bin->max_nz), bin->d_max, sizeof(int), cudaMemcpyDeviceToHost);
    if (bin->max_nz > B_PWMIN) {
        cudaMemcpy(bin->bin_size, bin->d_bin_size, sizeof(int) * BIN_NUM, cudaMemcpyDeviceToHost);
        cudaMemcpy(bin->d_bin_size, bin->bin_offset, sizeof(int) * BIN_NUM, cudaMemcpyHostToDevice);

        for (i = 0; i < BIN_NUM - 1; i++) {
            bin->bin_offset[i + 1] = bin->bin_offset[i] + bin->bin_size[i];
        }
        cudaMemcpy(bin->d_bin_offset, bin->bin_offset, sizeof(int) * BIN_NUM, cudaMemcpyHostToDevice);
  
        set_row_perm<<<GS, BS>>>(bin->d_bin_size, bin->d_bin_offset, bin->d_row_nz, bin->d_row_perm, M, B_MIN, B_PWMIN);
    }

    else {
        bin->bin_size[0] = M;
        for (i = 1; i < BIN_NUM; i++) {
            bin->bin_size[i] = 0;
        }
        bin->bin_offset[0] = 0;
        for (i = 1; i < BIN_NUM; i++) {
            bin->bin_offset[i] = M;
        }
        BS = 1024;
        GS = div_round_up(M, BS);
        init_row_perm<<<GS, BS>>>(bin->d_row_perm, M);
    }
}

__global__ void init_value(real *d_val, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nz) {
        return;
    }
    d_val[i] = 0;
}

__global__ void init_check(int *d_check, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nz) {
        return;
    }
    d_check[i] = -1;
}

__global__ void set_row_nz_bin_pwarp(const int *d_arpt, const int *d_acol,
                                     const int* __restrict__ d_brpt,
                                     const int* __restrict__ d_bcol,
                                     const int *d_row_perm,
                                     int *d_row_nz,
                                     int bin_offset, int M) {
  
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int rid = i / PWARP;
    int tid = i % PWARP;
    int local_rid = rid % (blockDim.x / PWARP);
  
    int j, k;
    int soffset;
    int acol, bcol, key, hash, adr, nz, old;
    __shared__ int check[IMB_PW_SH_SIZE];
  
    soffset = local_rid * IMB_PWMIN;
  
    for (j = tid; j < IMB_PWMIN; j += PWARP) {
        check[soffset + j] = -1;
    }
    if (rid >= M) {
        return;
    }

    rid = d_row_perm[rid + bin_offset];
    nz = 0;
    for (j = d_arpt[rid] + tid; j < d_arpt[rid + 1]; j += PWARP) {
        acol = ld_gbl_int32(d_acol + j);
        for (k = d_brpt[acol]; k < d_brpt[acol + 1]; k++) {
            bcol = d_bcol[k];
            key = bcol;
            hash = (bcol * HASH_SCAL) & (IMB_PWMIN - 1);
            adr = soffset + hash;
            while (1) {
                if (check[adr] == key) {
                    break;
                }
                else if (check[adr] == -1) {
                    old = atomicCAS(check + adr, -1, key);
                    if (old == -1) {
                        nz++;
                        break;
                    }
                }
                else {
                    hash = (hash + 1) & (IMB_PWMIN - 1);
                    adr = soffset + hash;
                }
            }
        }
    }
  
    for (j = PWARP / 2; j >= 1; j /= 2) {
        nz += __shfl_xor(nz, j);
    }

    if (tid == 0) {
        d_row_nz[rid] = nz;
    }
}


template <int SH_ROW>
__global__ void set_row_nz_bin_each(const int *d_arpt, const int *d_acol,
                                    const int* __restrict__ d_brpt,
                                    const int* __restrict__ d_bcol,
                                    const int *d_row_perm,
                                    int *d_row_nz, int bin_offset, int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int rid = i / WARP;
    int tid = i % WARP;
    int wid = rid % (blockDim.x / WARP);
    int j, k, l;
    int bcol, key, hash, old;
    int nz, adr;
    int acol, ccol;
    int soffset;

    soffset = wid * SH_ROW;

    __shared__ int check[IMB_SH_SIZE];
    for (j = tid; j < SH_ROW; j += WARP) {
        check[soffset + j] = -1;
    }
  
    if (rid >= M) {
        return;
    }
  
    acol = 0;
    nz = 0;
    rid = d_row_perm[rid + bin_offset];
    for (j = d_arpt[rid]; j < d_arpt[rid + 1]; j += WARP) {
        if (j + tid < d_arpt[rid + 1]) acol = ld_gbl_int32(d_acol + j + tid);
        for (l = 0; l < WARP && j + l < d_arpt[rid + 1]; l++) {
            ccol = __shfl(acol, l);
            for (k = d_brpt[ccol] + tid; k < d_brpt[ccol + 1]; k += WARP) {
                bcol = d_bcol[k];
                key = bcol;
                hash = (bcol * HASH_SCAL) & (SH_ROW - 1);
                adr = soffset + hash;
                while (1) {
                    if (check[adr] == key) {
                        break;
                    }
                    else if (check[adr] == -1) {
                        old = atomicCAS(check + adr, -1, key);
                        if (old == -1) {
                            nz++;
                            break;
                        }
                    }
                    else {
                        hash = (hash + 1) & (SH_ROW - 1);
                        adr = soffset + hash;
                    }
                }
            }
        }
    }

    for (j = WARP / 2; j >= 1; j /= 2) {
        nz += __shfl_xor(nz, j);
    }

    if (tid == 0) {
        d_row_nz[rid] = nz;
    }
}

template <int SH_ROW>
__global__ void set_row_nz_bin_each_tb(const int *d_arpt, const int *d_acol,
                                       const int* __restrict__ d_brpt,
                                       const int* __restrict__ d_bcol,
                                       int *d_row_perm, int *d_row_nz,
                                       int bin_offset, int M)
{
    int rid = blockIdx.x;
    int tid = threadIdx.x & (WARP - 1);
    int wid = threadIdx.x / WARP;
    int wnum = blockDim.x / WARP;
    int j, k;
    int bcol, key, hash, old;
    int nz, adr;
    int acol;

    __shared__ int check[SH_ROW];
    for (j = threadIdx.x; j < SH_ROW; j += blockDim.x) {
        check[j] = -1;
    }
  
    if (rid >= M) {
        return;
    }
  
    __syncthreads();

    nz = 0;
    rid = d_row_perm[rid + bin_offset];
    for (j = d_arpt[rid] + wid; j < d_arpt[rid + 1]; j += wnum) {
        acol = ld_gbl_int32(d_acol + j);
        for (k = d_brpt[acol] + tid; k < d_brpt[acol + 1]; k += WARP) {
            bcol = d_bcol[k];
            key = bcol;
            hash = (bcol * HASH_SCAL) & (SH_ROW - 1);
            adr = hash;
            while (1) {
                if (check[adr] == key) {
                    break;
                }
                else if (check[adr] == -1) {
                    old = atomicCAS(check + adr, -1, key);
                    if (old == -1) {
                        nz++;
                        break;
                    }
                }
                else {
                    hash = (hash + 1) & (SH_ROW - 1);
                    adr = hash;
                }
            }
        }
    }

    for (j = WARP / 2; j >= 1; j /= 2) {
        nz += __shfl_xor(nz, j);
    }
  
    __syncthreads();
    if (threadIdx.x == 0) {
        check[0] = 0;
    }
    __syncthreads();

    if (tid == 0) {
        atomicAdd(check, nz);
    }
    __syncthreads();
  
    if (threadIdx.x == 0) {
        d_row_nz[rid] = check[0];
    }
}

template <int SH_ROW>
__global__ void set_row_nz_bin_each_tb_large(const int *d_arpt, const int *d_acol,
                                             const int* __restrict__ d_brpt,
                                             const int* __restrict__ d_bcol,
                                             int *d_row_perm, int *d_row_nz,
                                             int *d_fail_count, int *d_fail_perm,
                                             int bin_offset, int M)
{
    int rid = blockIdx.x;
    int tid = threadIdx.x & (WARP - 1);
    int wid = threadIdx.x / WARP;
    int wnum = blockDim.x / WARP;
    int j, k;
    int bcol, key, hash, old;
    int adr;
    int acol;

    __shared__ int check[SH_ROW];
    __shared__ int snz[1];
    for (j = threadIdx.x; j < SH_ROW; j += blockDim.x) {
        check[j] = -1;
    }
    if (threadIdx.x == 0) {
        snz[0] = 0;
    }
  
    if (rid >= M) {
        return;
    }
  
    __syncthreads();
  
    rid = d_row_perm[rid + bin_offset];
    int count = 0;
    int border = SH_ROW >> 1;
    for (j = d_arpt[rid] + wid; j < d_arpt[rid + 1]; j += wnum) {
        acol = ld_gbl_int32(d_acol + j);
        for (k = d_brpt[acol] + tid; k < d_brpt[acol + 1]; k += WARP) {
            bcol = d_bcol[k];
            key = bcol;
            hash = (bcol * HASH_SCAL) & (SH_ROW - 1);
            adr = hash;
            while (count < border && snz[0] < border) {
                if (check[adr] == key) {
                    break;
                }
                else if (check[adr] == -1) {
                    old = atomicCAS(check + adr, -1, key);
                    if (old == -1) {
                        atomicAdd(snz, 1);
                        break;
                    }
                }
                else {
                    hash = (hash + 1) & (SH_ROW - 1);
                    adr = hash;
                    count++;
                }
            }
            if (count >= border || snz[0] >= border) {
                break;
            }
        }
        if (count >= border || snz[0] >= border) {
            break;
        }
    }
  
    __syncthreads();
    if (count >= border || snz[0] >= border) {
        if (threadIdx.x == 0) {
            int d = atomicAdd(d_fail_count, 1);
            d_fail_perm[d] = rid;
        }
    }
    else {
        if (threadIdx.x == 0) {
            d_row_nz[rid] = snz[0];
        }
    }
}

__global__ void set_row_nz_bin_each_gl(const int *d_arpt, const int *d_acol,
                                       const int* __restrict__ d_brpt,
                                       const int* __restrict__ d_bcol,
                                       const int *d_row_perm,
                                       int *d_row_nz, int *d_check,
                                       int max_row_nz, int bin_offset, int M)
{
    int rid = blockIdx.x;
    int tid = threadIdx.x & (WARP - 1);
    int wid = threadIdx.x / WARP;
    int wnum = blockDim.x / WARP;
    int j, k;
    int bcol, key, hash, old;
    int nz, adr;
    int acol;
    int offset = rid * max_row_nz;

    __shared__ int snz[1];
    if (threadIdx.x == 0) {
        snz[0] = 0;
    }
    __syncthreads();
  
    if (rid >= M) {
        return;
    }
  
    nz = 0;
    rid = d_row_perm[rid + bin_offset];
    for (j = d_arpt[rid] + wid; j < d_arpt[rid + 1]; j += wnum) {
        acol = ld_gbl_int32(d_acol + j);
        for (k = d_brpt[acol] + tid; k < d_brpt[acol + 1]; k += WARP) {
            bcol = d_bcol[k];
            key = bcol;
            hash = (bcol * HASH_SCAL) % max_row_nz;
            adr = offset + hash;
            while (1) {
                if (d_check[adr] == key) {
                    break;
                }
                else if (d_check[adr] == -1) {
                    old = atomicCAS(d_check + adr, -1, key);
                    if (old == -1) {
                        nz++;
                        break;
                    }
                }
                else {
                    hash = (hash + 1) % max_row_nz;
                    adr = offset + hash;
                }
            }
        }
    }
  
    for (j = WARP / 2; j >= 1; j /= 2) {
        nz += __shfl_xor(nz, j);
    }
  
    if (tid == 0) {
        atomicAdd(snz, nz);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        d_row_nz[rid] = snz[0];
    }
}

void set_row_nnz(int *d_arpt, int *d_acol,
                 int *d_brpt, int *d_bcol,
                 int *d_crpt,
                 sfBIN *bin,
                 int M, int *nnz);
  

__global__ void calculate_value_col_bin_pwarp(const int *d_arpt,
                                              const int *d_acol,
                                              const real *d_aval,
                                              const int* __restrict__ d_brpt,
                                              const int* __restrict__ d_bcol,
                                              const real* __restrict__ d_bval,
                                              int *d_crpt,
                                              int *d_ccol,
                                              real *d_cval,
                                              const int *d_row_perm,
                                              int *d_nz,
                                              int bin_offset,
                                              int bin_size) {
  
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int rid = i / PWARP;
    int tid = i % PWARP;
    int local_rid = rid % (blockDim.x / PWARP);
    int j;
    __shared__ int shared_check[B_PW_SH_SIZE];
    __shared__ real shared_value[B_PW_SH_SIZE];
  
    int soffset = local_rid * (B_PWMIN);
  
    for (j = tid; j < (B_PWMIN); j += PWARP) {
        shared_check[soffset + j] = -1;
        shared_value[soffset + j] = 0;
    }
  
    if (rid >= bin_size) {
        return;
    }
    rid = d_row_perm[rid + bin_offset];
  
    if (tid == 0) {
        d_nz[rid] = 0;
    }
    int k;
    int acol, bcol, hash, key, adr;
    int offset = d_crpt[rid];
    int old, index;
    real aval, bval;

    for (j = d_arpt[rid] + tid; j < d_arpt[rid + 1]; j += PWARP) {
        acol = ld_gbl_int32(d_acol + j);
        aval = ld_gbl_real(d_aval + j);
        for (k = d_brpt[acol]; k < d_brpt[acol + 1]; k++) {
            bcol = d_bcol[k];
            bval = d_bval[k];
	
            key = bcol;
            hash = (bcol * HASH_SCAL) & ((B_PWMIN) - 1);
            adr = soffset + hash;
            while (1) {
                if (shared_check[adr] == key) {
                    atomic_fadd(shared_value + adr, aval * bval);
                    break;
                }
                else if (shared_check[adr] == -1) {
                    old = atomicCAS(shared_check + adr, -1, key);
                    if (old == -1) {
                        atomic_fadd(shared_value + adr, aval * bval);
                        break;
                    }
                }
                else {
                    hash = (hash + 1) & ((B_PWMIN) - 1);
                    adr = soffset + hash;
                }
            }
        }
    }
  
    for (j = tid; j < (B_PWMIN); j += PWARP) {
        if (shared_check[soffset + j] != -1) {
            index = atomicAdd(d_nz + rid, 1);
            shared_check[soffset + index] = shared_check[soffset + j];
            shared_value[soffset + index] = shared_value[soffset + j];
        }
    }
    int nz = d_nz[rid];
    // Sorting for shared data
    int count, target;
    for (j = tid; j < nz; j += PWARP) {
        target = shared_check[soffset + j];
        count = 0;
        for (k = 0; k < nz; k++) {
            count += (unsigned int)(shared_check[soffset + k] - target) >> 31;
        }
        d_ccol[offset + count] = shared_check[soffset + j];
        d_cval[offset + count] = shared_value[soffset + j];
    }
}


template <int SH_ROW>
__global__ void calculate_value_col_bin_each(const int *d_arpt,
                                             const int *d_acol,
                                             const real *d_aval,
                                             const int* __restrict__ d_brpt,
                                             const int* __restrict__ d_bcol,
                                             const real* __restrict__ d_bval,
                                             int *d_crpt,
                                             int *d_ccol,
                                             real *d_cval,
                                             const int *d_row_perm,
                                             int *d_nz,
                                             int bin_offset,
                                             int bin_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int rid = i / WARP;
    int tid = i % WARP;
    int wid = rid % (blockDim.x / WARP);
    int j;
    __shared__ int shared_check[B_SH_SIZE];
    __shared__ real shared_value[B_SH_SIZE];
  
    int soffset = wid * SH_ROW;

    for (j = tid; j < SH_ROW; j += WARP) {
        shared_check[soffset + j] = -1;
        shared_value[soffset + j] = 0;
    }
  
    if (rid >= bin_size) {
        return;
    }
    rid = d_row_perm[rid + bin_offset];

    if (tid == 0) {
        d_nz[rid] = 0;
    }
    int lacol, acol;
    int k, l;
    int bcol, hash, key, adr;
    int offset = d_crpt[rid];
    int old, index;
    real laval, aval, bval;

    lacol = 0;
    for (j = d_arpt[rid]; j < d_arpt[rid + 1]; j += WARP) {
        if (j + tid < d_arpt[rid + 1]) {
            lacol = ld_gbl_int32(d_acol + j + tid);
            laval = ld_gbl_real(d_aval + j + tid);
        }
        for (l = 0; l < WARP && j + l < d_arpt[rid + 1]; l++) {
            acol = __shfl(lacol, l);
            aval = __shfl(laval, l);
            for (k = d_brpt[acol] + tid; k < d_brpt[acol + 1]; k += WARP) {
                bcol = d_bcol[k];
                bval = d_bval[k];
	
                key = bcol;
                hash = (bcol * HASH_SCAL) & (SH_ROW - 1);
                adr = soffset + hash;
                while (1) {
                    if (shared_check[adr] == key) {
                        atomic_fadd(shared_value + adr, aval * bval);
                        break;
                    }
                    else if (shared_check[adr] == -1) {
                        old = atomicCAS(shared_check + adr, -1, key);
                        if (old == -1) {
                            atomic_fadd(shared_value + adr, aval * bval);
                            break;
                        }
                    }
                    else {
                        hash = (hash + 1) & (SH_ROW - 1);
                        adr = soffset + hash;
                    }
                }
            }
        }
    }
  
    for (j = tid; j < SH_ROW; j += WARP) {
        if (shared_check[soffset + j] != -1) {
            index = atomicAdd(d_nz + rid, 1);
            shared_check[soffset + index] = shared_check[soffset + j];
            shared_value[soffset + index] = shared_value[soffset + j];
        }
    }
    int nz = d_nz[rid];
    /* Sorting for shared data */
    int count, target;
    for (j = tid; j < nz; j += WARP) {
        target = shared_check[soffset + j];
        count = 0;
        for (k = 0; k < nz; k++) {
            count += (unsigned int)(shared_check[soffset + k] - target) >> 31;
        }
        d_ccol[offset + count] = shared_check[soffset + j];
        d_cval[offset + count] = shared_value[soffset + j];
    }
}

template <int SH_ROW>
__global__ void calculate_value_col_bin_each_tb(const int *d_arpt,
                                                const int *d_acol,
                                                const real *d_aval,
                                                const int* __restrict__ d_brpt,
                                                const int* __restrict__ d_bcol,
                                                const real* __restrict__ d_bval,
                                                int *d_crpt,
                                                int *d_ccol,
                                                real *d_cval,
                                                const int *d_row_perm,
                                                int *d_nz,
                                                int bin_offset,
                                                int bin_size)
{
    int rid = blockIdx.x;
    int tid = threadIdx.x & (WARP - 1);
    int wid = threadIdx.x / WARP;
    int wnum = blockDim.x / WARP;
    int j;
    __shared__ int shared_check[SH_ROW];
    __shared__ real shared_value[SH_ROW];
  
    for (j = threadIdx.x; j < SH_ROW; j += blockDim.x) {
        shared_check[j] = -1;
        shared_value[j] = 0;
    }
  
    if (rid >= bin_size) {
        return;
    }

    rid = d_row_perm[rid + bin_offset];

    if (threadIdx.x == 0) {
        d_nz[rid] = 0;
    }
    __syncthreads();

    int acol;
    int k;
    int bcol, hash, key;
    int offset = d_crpt[rid];
    int old, index;
    real aval, bval;

    for (j = d_arpt[rid] + wid; j < d_arpt[rid + 1]; j += wnum) {
        acol = ld_gbl_int32(d_acol + j);
        aval = ld_gbl_real(d_aval + j);
        for (k = d_brpt[acol] + tid; k < d_brpt[acol + 1]; k += WARP) {
            bcol = d_bcol[k];
            bval = d_bval[k];
	
            key = bcol;
            hash = (bcol * HASH_SCAL) & (SH_ROW - 1);
            while (1) {
                if (shared_check[hash] == key) {
                    atomic_fadd(shared_value + hash, aval * bval);
                    break;
                }
                else if (shared_check[hash] == -1) {
                    old = atomicCAS(shared_check + hash, -1, key);
                    if (old == -1) {
                        atomic_fadd(shared_value + hash, aval * bval);
                        break;
                    }
                }
                else {
                    hash = (hash + 1) & (SH_ROW - 1);
                }
            }
        }
    }
  
    __syncthreads();
    if (threadIdx.x < WARP) {
        for (j = tid; j < SH_ROW; j += WARP) {
            if (shared_check[j] != -1) {
                index = atomicAdd(d_nz + rid, 1);
                shared_check[index] = shared_check[j];
                shared_value[index] = shared_value[j];
            }
        }
    }
    __syncthreads();
    int nz = d_nz[rid];
    /* Sorting for shared data */
    int count, target;
    for (j = threadIdx.x; j < nz; j += blockDim.x) {
        target = shared_check[j];
        count = 0;
        for (k = 0; k < nz; k++) {
            count += (unsigned int)(shared_check[k] - target) >> 31;
        }
        d_ccol[offset + count] = shared_check[j];
        d_cval[offset + count] = shared_value[j];
    }

}

__global__ void calculate_value_col_bin_each_gl(const int *d_arpt,
                                                const int *d_acol,
                                                const real *d_aval,
                                                const int* __restrict__ d_brpt,
                                                const int* __restrict__ d_bcol,
                                                const real* __restrict__ d_bval,
                                                int *d_crpt,
                                                int *d_ccol,
                                                real *d_cval,
                                                const int *d_row_perm,
                                                int *d_nz,
                                                int *d_check,
                                                real *d_value,
                                                int max_row_nz,
                                                int bin_offset,
                                                int M)
{
    int rid = blockIdx.x;
    int tid = threadIdx.x & (WARP - 1);
    int wid = threadIdx.x / WARP;
    int wnum = blockDim.x / WARP;
    int j;
  
    if (rid >= M) {
        return;
    }

    int doffset = rid * max_row_nz;

    rid = d_row_perm[rid + bin_offset];
  
    if (threadIdx.x == 0) {
        d_nz[rid] = 0;
    }
    __syncthreads();

    int acol;
    int k;
    int bcol, hash, key, adr;
    int offset = d_crpt[rid];
    int old, index;
    real aval, bval;

    for (j = d_arpt[rid] + wid; j < d_arpt[rid + 1]; j += wnum) {
        acol = ld_gbl_int32(d_acol + j);
        aval = ld_gbl_real(d_aval + j);
        for (k = d_brpt[acol] + tid; k < d_brpt[acol + 1]; k += WARP) {
            bcol = d_bcol[k];
            bval = d_bval[k];
      
            key = bcol;
            hash = (bcol * HASH_SCAL) % max_row_nz;
            adr = doffset + hash;
            while (1) {
                if (d_check[adr] == key) {
                    atomic_fadd(d_value + adr, aval * bval);
                    break;
                }
                else if (d_check[adr] == -1) {
                    old = atomicCAS(d_check + adr, -1, key);
                    if (old == -1) {
                        atomic_fadd(d_value + adr, aval * bval);
                        break;
                    }
                }
                else {
                    hash = (hash + 1) % max_row_nz;
                    adr = doffset + hash;
                }
            }
        }
    }
  
    __syncthreads();
    if (threadIdx.x < WARP) {
        for (j = tid; j < max_row_nz; j += WARP) {
            if (d_check[doffset + j] != -1) {
                index = atomicAdd(d_nz + rid, 1);
                d_check[doffset + index] = d_check[doffset + j];
                d_value[doffset + index] = d_value[doffset + j];
            }
        }
    }
    __syncthreads();
    int nz = d_nz[rid];
  
    /* Sorting for shared data */
    int count, target;
    for (j = threadIdx.x; j < nz; j += blockDim.x) {
        target = d_check[doffset + j];
        count = 0;
        for (k = 0; k < nz; k++) {
            count += (unsigned int)(d_check[doffset + k] - target) >> 31;
        }
        d_ccol[offset + count] = d_check[doffset + j];
        d_cval[offset + count] = d_value[doffset + j];
    }

}

void calculate_value_col_bin(int *d_arpt, int *d_acol, real *d_aval,
                             int *d_brpt, int *d_bcol, real *d_bval,
                             int *d_crpt, int *d_ccol, real *d_cval,
                             sfBIN *bin,
                             int M);
  
void spgemm_kernel_hash(sfCSR *a, sfCSR *b, sfCSR *c)
{
  
    int M;
    sfBIN bin;
  
    M = a->M;
    c->M = M;
    c->N = b->N;
  
    /* Initialize bin */
    init_bin(&bin, M);

    /* Set max bin */
    set_max_bin(a->d_rpt, a->d_col, b->d_rpt, &bin, M);
  
    checkCudaErrors(cudaMalloc((void **)&(c->d_rpt), sizeof(int) * (M + 1)));

    /* Count nz of C */
    set_row_nnz(a->d_rpt, a->d_col,
                b->d_rpt, b->d_col,
                c->d_rpt,
                &bin,
                M,
                &(c->nnz));

    /* Set bin */
    set_min_bin(&bin, M);
  
    checkCudaErrors(cudaMalloc((void **)&(c->d_col), sizeof(int) * c->nnz));
    checkCudaErrors(cudaMalloc((void **)&(c->d_val), sizeof(real) * c->nnz));
  
    /* Calculating value of C */
    calculate_value_col_bin(a->d_rpt, a->d_col, a->d_val,
                            b->d_rpt, b->d_col, b->d_val,
                            c->d_rpt, c->d_col, c->d_val,
                            &bin,
                            M);

    release_bin(bin);
}

