#include <iostream>

#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include <nsparse.hpp>
#include <nsparse_asm.hpp>
#include <CSR.hpp>
#include <BIN.hpp>

#ifndef SPGEMM_H
#define SPGEMM_H

#define BIN_NUM 7
#define PWARP 4
#define TS_S_P 32 //Table Size for Symbolic in PWARP per row
#define TS_N_P 16 //Table Size for Numeric in PWARP per row
#define TS_S_T 512 //Table Size for Symbolic in Thread block per row
#define TS_N_T 256 //Table Size for Numeric in Thread block per row

#define SHARED_S_P 4096
#define SHARED_N_P 2048
#define IMB_SH_SIZE 1024
#define B_SH_SIZE 512
#define HASH_SCAL 107

template <class idType>
__global__ void set_flop_per_row(idType *d_arpt, idType *d_acol, const idType* __restrict__ d_brpt, long long int *d_flop_per_row, idType nrow)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nrow) {
        return;
    }
    idType flop_per_row = 0;
    idType j;
    for (j = d_arpt[i]; j < d_arpt[i + 1]; j++) {
        flop_per_row += d_brpt[d_acol[j] + 1] - d_brpt[d_acol[j]];
    }
    d_flop_per_row[i] = flop_per_row;
}

template <class idType, class valType>
void get_spgemm_flop(CSR<idType, valType> a, CSR<idType, valType> b, long long int &flop)
{
    int GS, BS;
    long long int *d_flop_per_row;

    BS = MAX_LOCAL_THREAD_NUM;
    checkCudaErrors(cudaMalloc((void **)&(d_flop_per_row), sizeof(long long int) * a.nrow));
  
    GS = div_round_up(a.nrow, BS);
    set_flop_per_row<<<GS, BS>>>(a.d_rpt, a.d_colids, b.d_rpt, d_flop_per_row, a.nrow);
  
    long long int *tmp = (long long int *)malloc(sizeof(long long int) * a.nrow);
    cudaMemcpy(tmp, d_flop_per_row, sizeof(long long int) * a.nrow, cudaMemcpyDeviceToHost);
    flop = thrust::reduce(thrust::device, d_flop_per_row, d_flop_per_row + a.nrow);

    flop *= 2;
    cudaFree(d_flop_per_row);

}

template <class idType, class valType>
cusparseStatus_t SpGEMM_cuSPARSE_numeric(CSR<idType, valType> a, CSR<idType, valType> b, CSR<idType, valType> &c, cusparseHandle_t cusparseHandle, cusparseOperation_t trans_a, cusparseOperation_t trans_b, cusparseMatDescr_t descr_a, cusparseMatDescr_t descr_b, cusparseMatDescr_t descr_c);

template <>
cusparseStatus_t SpGEMM_cuSPARSE_numeric<int, float>(CSR<int, float> a, CSR<int, float> b, CSR<int, float> &c, cusparseHandle_t cusparseHandle, cusparseOperation_t trans_a, cusparseOperation_t trans_b, cusparseMatDescr_t descr_a, cusparseMatDescr_t descr_b, cusparseMatDescr_t descr_c)
{
    return cusparseScsrgemm(cusparseHandle, trans_a, trans_b, a.nrow, b.ncolumn, a.ncolumn, descr_a, a.nnz, a.d_values, a.d_rpt, a.d_colids, descr_b, b.nnz, b.d_values, b.d_rpt, b.d_colids, descr_c, c.d_values, c.d_rpt, c.d_colids);
}

template <>
cusparseStatus_t SpGEMM_cuSPARSE_numeric<int, double>(CSR<int, double> a, CSR<int, double> b, CSR<int, double> &c, cusparseHandle_t cusparseHandle, cusparseOperation_t trans_a, cusparseOperation_t trans_b, cusparseMatDescr_t descr_a, cusparseMatDescr_t descr_b, cusparseMatDescr_t descr_c)
{
    return cusparseDcsrgemm(cusparseHandle, trans_a, trans_b, a.nrow, b.ncolumn, a.ncolumn, descr_a, a.nnz, a.d_values, a.d_rpt, a.d_colids, descr_b, b.nnz, b.d_values, b.d_rpt, b.d_colids, descr_c, c.d_values, c.d_rpt, c.d_colids);
}

template <class idType, class valType>
void SpGEMM_cuSPARSE_kernel(CSR<idType, valType> a, CSR<idType, valType> b, CSR<idType, valType> &c, cusparseHandle_t cusparseHandle, cusparseOperation_t trans_a, cusparseOperation_t trans_b, cusparseMatDescr_t descr_a, cusparseMatDescr_t descr_b, cusparseMatDescr_t descr_c)
{
    cusparseStatus_t status;
    c.nrow = a.nrow;
    c.ncolumn = b.ncolumn;
    cudaMalloc((void **)&(c.d_rpt), sizeof(idType) * (c.nrow + 1));

    status = cusparseXcsrgemmNnz(cusparseHandle, trans_a, trans_b, a.nrow, b.ncolumn, a.ncolumn, descr_a, a.nnz, a.d_rpt, a.d_colids, descr_b, b.nnz, b.d_rpt, b.d_colids, descr_c, c.d_rpt, &(c.nnz));
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cout << "cuSPARSE failed at Symbolic phase" << endl;
    }

    cudaMalloc((void **)&(c.d_colids), sizeof(idType) * (c.nnz));
    cudaMalloc((void **)&(c.d_values), sizeof(valType) * (c.nnz));
        
    status = SpGEMM_cuSPARSE_numeric(a, b, c, cusparseHandle, trans_a, trans_b, descr_a, descr_b, descr_c);
    
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cout << "cuSPARSE failed at Numeric phase" << endl;
    }
}

template <class idType, class valType>
void SpGEMM_cuSPARSE(CSR<idType, valType> a, CSR<idType, valType> b, CSR<idType, valType> &c)
{
    cusparseHandle_t cusparseHandle;
    cusparseMatDescr_t descr_a, descr_b, descr_c;
    cusparseOperation_t trans_a, trans_b;

    trans_a = trans_b = CUSPARSE_OPERATION_NON_TRANSPOSE;
  
    /* Set up cuSPARSE Library */
    cusparseCreate(&cusparseHandle);
    cusparseCreateMatDescr(&descr_a);
    cusparseCreateMatDescr(&descr_b);
    cusparseCreateMatDescr(&descr_c);
    cusparseSetMatType(descr_a, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatType(descr_b, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatType(descr_c, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr_a, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatIndexBase(descr_b, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatIndexBase(descr_c, CUSPARSE_INDEX_BASE_ZERO);
  
    /* Execution of SpMV on Device */
    SpGEMM_cuSPARSE_kernel(a, b, c,
                           cusparseHandle,
                           trans_a, trans_b,
                           descr_a, descr_b, descr_c);
    cudaThreadSynchronize();
    
    c.memcpyDtH();

    c.release_csr();
    cusparseDestroy(cusparseHandle);
}

template <class idType, class valType>
__global__ void init_value_table(valType *d_values, idType nnz)
{
    idType i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnz) {
        return;
    }
    d_values[i] = 0;
}

template <class idType>
__global__ void init_id_table(idType *d_id_table, idType nnz)
{
    idType i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnz) {
        return;
    }
    d_id_table[i] = -1;
}

template <class idType>
__global__ void hash_symbolic_pwarp(const idType *d_arpt, const idType *d_acolids,
                                    const idType* __restrict__ d_brpt,
                                    const idType* __restrict__ d_bcolids,
                                    const idType *d_permutation,
                                    idType *d_row_nz,
                                    idType bin_offset, idType M)
{
    idType i = blockIdx.x * blockDim.x + threadIdx.x;
    idType rid = i / PWARP;
    idType tid = i % PWARP;
    idType local_rid = rid % (blockDim.x / PWARP);
  
    idType j, k;
    idType soffset;
    idType acol, bcol, key, hash, adr, nz, old;
    __shared__ idType id_table[SHARED_S_P];
  
    soffset = local_rid * TS_S_P;
  
    for (j = tid; j < TS_S_P; j += PWARP) {
        id_table[soffset + j] = -1;
    }
    if (rid >= M) {
        return;
    }

    rid = d_permutation[rid + bin_offset];
    nz = 0;
    for (j = d_arpt[rid] + tid; j < d_arpt[rid + 1]; j += PWARP) {
        acol = ld_gbl_col(d_acolids + j);
        for (k = d_brpt[acol]; k < d_brpt[acol + 1]; k++) {
            bcol = d_bcolids[k];
            key = bcol;
            hash = (bcol * HASH_SCAL) & (TS_S_P - 1);
            adr = soffset + hash;
            while (1) {
                if (id_table[adr] == key) {
                    break;
                }
                else if (id_table[adr] == -1) {
                    old = atomicCAS(id_table + adr, -1, key);
                    if (old == -1) {
                        nz++;
                        break;
                    }
                }
                else {
                    hash = (hash + 1) & (TS_S_P - 1);
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

template <class idType, int SH_ROW>
__global__ void hash_symbolic_tb(const idType *d_arpt, const idType *d_acolids,
                                 const idType* __restrict__ d_brpt,
                                 const idType* __restrict__ d_bcolids,
                                 idType *d_permutation, idType *d_row_nz,
                                 idType bin_offset, idType M)
{
    idType rid = blockIdx.x;
    idType tid = threadIdx.x & (warp - 1);
    idType wid = threadIdx.x / warp;
    idType wnum = blockDim.x / warp;
    idType j, k;
    idType bcol, key, hash, old;
    idType nz, adr;
    idType acol;

    __shared__ idType id_table[SH_ROW];
    for (j = threadIdx.x; j < SH_ROW; j += blockDim.x) {
        id_table[j] = -1;
    }
  
    if (rid >= M) {
        return;
    }
  
    __syncthreads();

    nz = 0;
    rid = d_permutation[rid + bin_offset];
    for (j = d_arpt[rid] + wid; j < d_arpt[rid + 1]; j += wnum) {
        acol = ld_gbl_col(d_acolids + j);
        for (k = d_brpt[acol] + tid; k < d_brpt[acol + 1]; k += warp) {
            bcol = d_bcolids[k];
            key = bcol;
            hash = (bcol * HASH_SCAL) & (SH_ROW - 1);
            adr = hash;
            while (1) {
                if (id_table[adr] == key) {
                    break;
                }
                else if (id_table[adr] == -1) {
                    old = atomicCAS(id_table + adr, -1, key);
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

    for (j = warp / 2; j >= 1; j /= 2) {
        nz += __shfl_xor(nz, j);
    }
  
    __syncthreads();
    if (threadIdx.x == 0) {
        id_table[0] = 0;
    }
    __syncthreads();

    if (tid == 0) {
        atomicAdd(id_table, nz);
    }
    __syncthreads();
  
    if (threadIdx.x == 0) {
        d_row_nz[rid] = id_table[0];
    }
}

template <class idType, int SH_ROW>
__global__ void hash_symbolic_tb_large(const idType *d_arpt, const idType *d_acolids,
                                       const idType* __restrict__ d_brpt,
                                       const idType* __restrict__ d_bcolids,
                                       idType *d_permutation, idType *d_row_nz,
                                       idType *d_fail_count, idType *d_fail_perm,
                                       idType bin_offset, idType M)
{
    idType rid = blockIdx.x;
    idType tid = threadIdx.x & (warp - 1);
    idType wid = threadIdx.x / warp;
    idType wnum = blockDim.x / warp;
    idType j, k;
    idType bcol, key, hash, old;
    idType adr;
    idType acol;

    __shared__ idType id_table[SH_ROW];
    __shared__ idType snz[1];
    for (j = threadIdx.x; j < SH_ROW; j += blockDim.x) {
        id_table[j] = -1;
    }
    if (threadIdx.x == 0) {
        snz[0] = 0;
    }
  
    if (rid >= M) {
        return;
    }
  
    __syncthreads();
  
    rid = d_permutation[rid + bin_offset];
    idType count = 0;
    idType border = SH_ROW >> 1;
    for (j = d_arpt[rid] + wid; j < d_arpt[rid + 1]; j += wnum) {
        acol = ld_gbl_col(d_acolids + j);
        for (k = d_brpt[acol] + tid; k < d_brpt[acol + 1]; k += warp) {
            bcol = d_bcolids[k];
            key = bcol;
            hash = (bcol * HASH_SCAL) & (SH_ROW - 1);
            adr = hash;
            while (count < border && snz[0] < border) {
                if (id_table[adr] == key) {
                    break;
                }
                else if (id_table[adr] == -1) {
                    old = atomicCAS(id_table + adr, -1, key);
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

template <class idType>
__global__ void hash_symbolic_gl(const idType *d_arpt, const idType *d_acol,
                                 const idType* __restrict__ d_brpt,
                                 const idType* __restrict__ d_bcol,
                                 const idType *d_permutation,
                                 idType *d_row_nz, idType *d_id_table,
                                 idType max_row_nz, idType bin_offset, idType M)
{
    idType rid = blockIdx.x;
    idType tid = threadIdx.x & (warp - 1);
    idType wid = threadIdx.x / warp;
    idType wnum = blockDim.x / warp;
    idType j, k;
    idType bcol, key, hash, old;
    idType nz, adr;
    idType acol;
    idType offset = rid * max_row_nz;

    __shared__ idType snz[1];
    if (threadIdx.x == 0) {
        snz[0] = 0;
    }
    __syncthreads();
  
    if (rid >= M) {
        return;
    }
  
    nz = 0;
    rid = d_permutation[rid + bin_offset];
    for (j = d_arpt[rid] + wid; j < d_arpt[rid + 1]; j += wnum) {
        acol = ld_gbl_col(d_acol + j);
        for (k = d_brpt[acol] + tid; k < d_brpt[acol + 1]; k += warp) {
            bcol = d_bcol[k];
            key = bcol;
            hash = (bcol * HASH_SCAL) % max_row_nz;
            adr = offset + hash;
            while (1) {
                if (d_id_table[adr] == key) {
                    break;
                }
                else if (d_id_table[adr] == -1) {
                    old = atomicCAS(d_id_table + adr, -1, key);
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
  
    for (j = warp / 2; j >= 1; j /= 2) {
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

template <class idType, class valType>
void hash_symbolic(CSR<idType, valType> a, CSR<idType, valType> b, CSR<idType, valType> &c, BIN<idType, BIN_NUM> &bin)
{
    int i;
    int GS, BS;
    for (i = BIN_NUM - 1; i >= 0; i--) {
        if (bin.bin_size[i] > 0) {
            switch (i) {
            case 0 :
                BS = 512;
                GS = div_round_up(bin.bin_size[i] * PWARP, BS);
                hash_symbolic_pwarp<<<GS, BS, 0, bin.stream[i]>>>(a.d_rpt, a.d_colids, b.d_rpt, b.d_colids, bin.d_permutation, bin.d_count, bin.bin_offset[i], bin.bin_size[i]);
                break;
            case 1 :
                BS = 64;
                GS = bin.bin_size[i];
            	hash_symbolic_tb<idType, 512><<<GS, BS, 0, bin.stream[i]>>>(a.d_rpt, a.d_colids, b.d_rpt, b.d_colids, bin.d_permutation, bin.d_count, bin.bin_offset[i], bin.bin_size[i]);
            	break;
            case 2 :
                BS = 128;
                GS = bin.bin_size[i];
            	hash_symbolic_tb<idType, 1024><<<GS, BS, 0, bin.stream[i]>>>(a.d_rpt, a.d_colids, b.d_rpt, b.d_colids, bin.d_permutation, bin.d_count, bin.bin_offset[i], bin.bin_size[i]);
            	break;
            case 3 :
                BS = 256;
                GS = bin.bin_size[i];
            	hash_symbolic_tb<idType, 2048><<<GS, BS, 0, bin.stream[i]>>>(a.d_rpt, a.d_colids, b.d_rpt, b.d_colids, bin.d_permutation, bin.d_count, bin.bin_offset[i], bin.bin_size[i]);
            	break;
            case 4 :
                BS = 512;
                GS = bin.bin_size[i];
            	hash_symbolic_tb<idType, 4096><<<GS, BS, 0, bin.stream[i]>>>(a.d_rpt, a.d_colids, b.d_rpt, b.d_colids, bin.d_permutation, bin.d_count, bin.bin_offset[i], bin.bin_size[i]);
            	break;
            case 5 :
                BS = 1024;
                GS = bin.bin_size[i];
            	hash_symbolic_tb<idType, 8192><<<GS, BS, 0, bin.stream[i]>>>(a.d_rpt, a.d_colids, b.d_rpt, b.d_colids, bin.d_permutation, bin.d_count, bin.bin_offset[i], bin.bin_size[i]);
            	break;
            case 6 :
                {
            	    idType fail_count;
            	    idType *d_fail_count, *d_fail_perm;
            	    fail_count = 0;
            	    checkCudaErrors(cudaMalloc((void **)&d_fail_count, sizeof(idType)));
            	    checkCudaErrors(cudaMalloc((void **)&d_fail_perm, sizeof(idType) * bin.bin_size[i]));
            	    cudaMemcpy(d_fail_count, &fail_count, sizeof(idType), cudaMemcpyHostToDevice);
            	    BS = 1024;
            	    GS = bin.bin_size[i];
            	    hash_symbolic_tb_large<idType, 8192><<<GS, BS, 0, bin.stream[i]>>>(a.d_rpt, a.d_colids, b.d_rpt, b.d_colids, bin.d_permutation, bin.d_count, d_fail_count, d_fail_perm, bin.bin_offset[i], bin.bin_size[i]);
            	    cudaMemcpy(&fail_count, d_fail_count, sizeof(idType), cudaMemcpyDeviceToHost);
            	    if (fail_count > 0) {
              	        idType max_row_nz = bin.max_flop;
            	        size_t table_size = (size_t)max_row_nz * fail_count;
            	        idType *d_id_table;
            	        checkCudaErrors(cudaMalloc((void **)&(d_id_table), sizeof(idType) * table_size));
            	        BS = 1024;
            	        GS = div_round_up(table_size, BS);
            	        init_id_table<idType><<<GS, BS, 0, bin.stream[i]>>>(d_id_table, table_size);
            	        GS = bin.bin_size[i];
	                    hash_symbolic_gl<idType><<<GS, BS, 0, bin.stream[i]>>>(a.d_rpt, a.d_colids, b.d_rpt, b.d_colids, d_fail_perm, bin.d_count, d_id_table, max_row_nz, 0, fail_count);
                        cudaFree(d_id_table);
  	                }
                    cudaFree(d_fail_count);
                    cudaFree(d_fail_perm);
                }
                break;
            default:
                ;
            }
        }
    }
    cudaThreadSynchronize();
    thrust::exclusive_scan(thrust::device, bin.d_count, bin.d_count + (a.nrow + 1), c.d_rpt, 0);
    cudaMemcpy(&(c.nnz), c.d_rpt + c.nrow, sizeof(idType), cudaMemcpyDeviceToHost);
    cout << c.nnz << endl;
}

template <class idType, class valType>
__global__ void hash_numeric_pwarp(const idType *d_arpt, const idType *d_acol, const valType *d_aval, const idType* __restrict__ d_brpt, const idType* __restrict__ d_bcol, const valType* __restrict__ d_bval, idType *d_crpt, idType *d_ccol, valType *d_cval, const idType *d_permutation, idType *d_nz, idType bin_offset, idType bin_size)
{
    idType i = blockIdx.x * blockDim.x + threadIdx.x;
    int rid = i / PWARP;
    int tid = i % PWARP;
    int local_rid = rid % (blockDim.x / PWARP);
    int j;

    __shared__ idType id_table[SHARED_N_P];
    __shared__ valType value_table[SHARED_N_P];
  
    idType soffset = local_rid * (TS_N_P);
  
    for (j = tid; j < TS_N_P; j += PWARP) {
        id_table[soffset + j] = -1;
        value_table[soffset + j] = 0;
    }
  
    if (rid >= bin_size) {
        return;
    }
    rid = d_permutation[rid + bin_offset];
  
    if (tid == 0) {
        d_nz[rid] = 0;
    }

    idType k;
    idType acol, bcol, hash, key, adr;
    idType offset = d_crpt[rid];
    idType old, index;
    valType aval, bval;

    for (j = d_arpt[rid] + tid; j < d_arpt[rid + 1]; j += PWARP) {
        acol = ld_gbl_col(d_acol + j);
        aval = ld_gbl_val(d_aval + j);
        for (k = d_brpt[acol]; k < d_brpt[acol + 1]; k++) {
            bcol = d_bcol[k];
            bval = d_bval[k];
            
            key = bcol;
            hash = (bcol * HASH_SCAL) & ((TS_N_P) - 1);
            adr = soffset + hash;
            while (1) {
                if (id_table[adr] == key) {
                    atomicAdd(value_table + adr, aval * bval);
                    break;
                }
                else if (id_table[adr] == -1) {
                    old = atomicCAS(id_table + adr, -1, key);
                    if (old == -1) {
                        atomicAdd(value_table + adr, aval * bval);
                        break;
                    }
                }
                else {
                    hash = (hash + 1) & ((TS_N_P) - 1);
                    adr = soffset + hash;
                }
            }
        }
    }
  
    for (j = tid; j < (TS_N_P); j += PWARP) {
        if (id_table[soffset + j] != -1) {
            index = atomicAdd(d_nz + rid, 1);
            id_table[soffset + index] = id_table[soffset + j];
            value_table[soffset + index] = value_table[soffset + j];
        }
    }
    
    idType nz = d_nz[rid];
    // Sorting for shared data
    idType count, target;
    for (j = tid; j < nz; j += PWARP) {
        target = id_table[soffset + j];
        count = 0;
        for (k = 0; k < nz; k++) {
            count += (unsigned int)(id_table[soffset + k] - target) >> 31;
        }
        d_ccol[offset + count] = id_table[soffset + j];
        d_cval[offset + count] = value_table[soffset + j];
    }
}

template <class idType, class valType, int SH_ROW>
__global__ void hash_numeric_tb(const idType *d_arpt, const idType *d_acolids, const valType *d_avalues, const idType* __restrict__ d_brpt, const idType* __restrict__ d_bcolids, const valType* __restrict__ d_bvalues, idType *d_crpt, idType *d_ccolids, valType *d_cvalues, const idType *d_permutation, idType *d_nz, idType bin_offset, idType bin_size)
{
    idType rid = blockIdx.x;
    idType tid = threadIdx.x & (warp - 1);
    idType wid = threadIdx.x / warp;
    idType wnum = blockDim.x / warp;
    idType j;
    __shared__ idType id_table[SH_ROW];
    __shared__ valType value_table[SH_ROW];
  
    for (j = threadIdx.x; j < SH_ROW; j += blockDim.x) {
        id_table[j] = -1;
        value_table[j] = 0;
    }
  
    if (rid >= bin_size) {
        return;
    }

    rid = d_permutation[rid + bin_offset];

    if (threadIdx.x == 0) {
        d_nz[rid] = 0;
    }
    __syncthreads();

    idType acolids;
    idType k;
    idType bcolids, hash, key;
    idType offset = d_crpt[rid];
    idType old, index;
    valType avalues, bvalues;

    for (j = d_arpt[rid] + wid; j < d_arpt[rid + 1]; j += wnum) {
        acolids = ld_gbl_col(d_acolids + j);
        avalues = ld_gbl_val(d_avalues + j);
        for (k = d_brpt[acolids] + tid; k < d_brpt[acolids + 1]; k += warp) {
            bcolids = d_bcolids[k];
            bvalues = d_bvalues[k];
	
            key = bcolids;
            hash = (bcolids * HASH_SCAL) & (SH_ROW - 1);
            while (1) {
                if (id_table[hash] == key) {
                    atomicAdd(value_table + hash, avalues * bvalues);
                    break;
                }
                else if (id_table[hash] == -1) {
                    old = atomicCAS(id_table + hash, -1, key);
                    if (old == -1) {
                        atomicAdd(value_table + hash, avalues * bvalues);
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
    if (threadIdx.x < warp) {
        for (j = tid; j < SH_ROW; j += warp) {
            if (id_table[j] != -1) {
                index = atomicAdd(d_nz + rid, 1);
                id_table[index] = id_table[j];
                value_table[index] = value_table[j];
            }
        }
    }
    __syncthreads();
    idType nz = d_nz[rid];
    /* Sorting for shared data */
    idType count, target;
    for (j = threadIdx.x; j < nz; j += blockDim.x) {
        target = id_table[j];
        count = 0;
        for (k = 0; k < nz; k++) {
            count += (unsigned int)(id_table[k] - target) >> 31;
        }
        d_ccolids[offset + count] = id_table[j];
        d_cvalues[offset + count] = value_table[j];
    }
}

template <class idType, class valType>
__global__ void hash_numeric_gl(const idType *d_arpt, const idType *d_acolids, const valType *d_avalues, const idType* __restrict__ d_brpt, const idType* __restrict__ d_bcolids, const valType* __restrict__ d_bvalues, idType *d_crpt, idType *d_ccolids, valType *d_cvalues, const idType *d_permutation, idType *d_nz, idType *d_id_table, valType *d_value_table, idType max_row_nz, idType bin_offset, idType M)
{
    idType rid = blockIdx.x;
    idType tid = threadIdx.x & (warp - 1);
    idType wid = threadIdx.x / warp;
    idType wnum = blockDim.x / warp;
    idType j;
  
    if (rid >= M) {
        return;
    }

    idType doffset = rid * max_row_nz;

    rid = d_permutation[rid + bin_offset];
  
    if (threadIdx.x == 0) {
        d_nz[rid] = 0;
    }
    __syncthreads();

    idType acolids;
    idType k;
    idType bcolids, hash, key, adr;
    idType offset = d_crpt[rid];
    idType old, index;
    valType avalues, bvalues;

    for (j = d_arpt[rid] + wid; j < d_arpt[rid + 1]; j += wnum) {
        acolids = ld_gbl_col(d_acolids + j);
        avalues = ld_gbl_val(d_avalues + j);
        for (k = d_brpt[acolids] + tid; k < d_brpt[acolids + 1]; k += warp) {
            bcolids = d_bcolids[k];
            bvalues = d_bvalues[k];
      
            key = bcolids;
            hash = (bcolids * HASH_SCAL) % max_row_nz;
            adr = doffset + hash;
            while (1) {
                if (d_id_table[adr] == key) {
                    atomicAdd(d_value_table + adr, avalues * bvalues);
                    break;
                }
                else if (d_id_table[adr] == -1) {
                    old = atomicCAS(d_id_table + adr, -1, key);
                    if (old == -1) {
                        atomicAdd(d_value_table + adr, avalues * bvalues);
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
    if (threadIdx.x < warp) {
        for (j = tid; j < max_row_nz; j += warp) {
            if (d_id_table[doffset + j] != -1) {
                index = atomicAdd(d_nz + rid, 1);
                d_id_table[doffset + index] = d_id_table[doffset + j];
                d_value_table[doffset + index] = d_value_table[doffset + j];
            }
        }
    }
    __syncthreads();
    idType nz = d_nz[rid];
  
    /* Sorting for shared data */
    idType count, target;
    for (j = threadIdx.x; j < nz; j += blockDim.x) {
        target = d_id_table[doffset + j];
        count = 0;
        for (k = 0; k < nz; k++) {
            count += (unsigned int)(d_id_table[doffset + k] - target) >> 31;
        }
        d_ccolids[offset + count] = d_id_table[doffset + j];
        d_cvalues[offset + count] = d_value_table[doffset + j];
    }

}

template <class idType, class valType>
void hash_numeric(CSR<idType, valType> a, CSR<idType, valType> b, CSR<idType, valType> &c, BIN<idType, BIN_NUM> &bin)
{
    int i;
    int GS, BS;
    for (i = BIN_NUM - 1; i >= 0; i--) {
        if (bin.bin_size[i] > 0) {
            switch (i) {
            case 0:
                BS = 512;
                GS = div_round_up(bin.bin_size[i] * PWARP, BS);
                hash_numeric_pwarp<idType, valType><<<GS, BS, 0, bin.stream[i]>>>(a.d_rpt, a.d_colids, a.d_values, b.d_rpt, b.d_colids, b.d_values, c.d_rpt, c.d_colids, c.d_values, bin.d_permutation, bin.d_count, bin.bin_offset[i], bin.bin_size[i]);
                break;
            case 1:
                BS = 64;
                GS = bin.bin_size[i];
                hash_numeric_tb<idType, valType, 256><<<GS, BS, 0, bin.stream[i]>>>(a.d_rpt, a.d_colids, a.d_values, b.d_rpt, b.d_colids, b.d_values, c.d_rpt, c.d_colids, c.d_values, bin.d_permutation, bin.d_count, bin.bin_offset[i], bin.bin_size[i]);
                break;
            case 2:
                BS = 128;
                GS = bin.bin_size[i];
                hash_numeric_tb<idType, valType, 512><<<GS, BS, 0, bin.stream[i]>>>(a.d_rpt, a.d_colids, a.d_values, b.d_rpt, b.d_colids, b.d_values, c.d_rpt, c.d_colids, c.d_values, bin.d_permutation, bin.d_count, bin.bin_offset[i], bin.bin_size[i]);
                break;
            case 3:
                BS = 256;
                GS = bin.bin_size[i];
                hash_numeric_tb<idType, valType, 1024><<<GS, BS, 0, bin.stream[i]>>>(a.d_rpt, a.d_colids, a.d_values, b.d_rpt, b.d_colids, b.d_values, c.d_rpt, c.d_colids, c.d_values, bin.d_permutation, bin.d_count, bin.bin_offset[i], bin.bin_size[i]);
                break;
            case 4:
                BS = 512;
                GS = bin.bin_size[i];
                hash_numeric_tb<idType, valType, 2048><<<GS, BS, 0, bin.stream[i]>>>(a.d_rpt, a.d_colids, a.d_values, b.d_rpt, b.d_colids, b.d_values, c.d_rpt, c.d_colids, c.d_values, bin.d_permutation, bin.d_count, bin.bin_offset[i], bin.bin_size[i]);
                break;
            case 5:
                BS = 1024;
                GS = bin.bin_size[i];
                hash_numeric_tb<idType, valType, 4096><<<GS, BS, 0, bin.stream[i]>>>(a.d_rpt, a.d_colids, a.d_values, b.d_rpt, b.d_colids, b.d_values, c.d_rpt, c.d_colids, c.d_values, bin.d_permutation, bin.d_count, bin.bin_offset[i], bin.bin_size[i]);
                break;
            case 6 :
                {
                    idType max_row_nz = bin.max_nz * 2;
                    idType table_size = max_row_nz * bin.bin_size[i];
                    idType *d_id_table;
                    valType *d_value_table;
                    checkCudaErrors(cudaMalloc((void **)&(d_id_table), sizeof(idType) * table_size));
                    checkCudaErrors(cudaMalloc((void **)&(d_value_table), sizeof(valType) * table_size));
                    BS = 1024;
                    GS = div_round_up(table_size, BS);
                    init_id_table<<<GS, BS, 0, bin.stream[i]>>>(d_id_table, table_size);
                    init_value_table<<<GS, BS, 0, bin.stream[i]>>>(d_value_table, table_size);
                    GS = bin.bin_size[i];
                    hash_numeric_gl<<<GS, BS, 0, bin.stream[i]>>>(a.d_rpt, a.d_colids, a.d_values, b.d_rpt, b.d_colids, b.d_values, c.d_rpt, c.d_colids, c.d_values, bin.d_permutation, bin.d_count, d_id_table, d_value_table, max_row_nz, bin.bin_offset[i], bin.bin_size[i]);
                    cudaFree(d_id_table);
                    cudaFree(d_value_table);
                }
                break;
            }
        }
    }
    cudaThreadSynchronize();
}

template <class idType, class valType>
void SpGEMM_Hash(CSR<idType, valType> a, CSR<idType, valType> b, CSR<idType, valType> &c)
{
    BIN<idType, BIN_NUM> bin(a.nrow);

    c.nrow = a.nrow;
    c.ncolumn = b.ncolumn;
    cudaMalloc((void **)&(c.d_rpt), sizeof(idType) * (c.nrow + 1));
    
    bin.set_max_bin(a.d_rpt, a.d_colids, b.d_rpt, a.nrow, TS_S_P, TS_S_T);
    hash_symbolic(a, b, c, bin);
    
    cudaMalloc((void **)&(c.d_colids), sizeof(idType) * (c.nnz));
    cudaMalloc((void **)&(c.d_values), sizeof(valType) * (c.nnz));

    bin.set_min_bin(a.nrow, TS_N_P, TS_N_T);
    hash_numeric(a, b, c, bin);
}



#endif

