#include <stdio.h>
#include <stdlib.h>
#include <time.h>
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

__global__ void set_intprod_per_row(int *d_arpt, int *d_acol,
                                    const int* __restrict__ d_brpt,
                                    long long int *d_max_row_nz,
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
    d_max_row_nz[i] = nz_per_row;
}

void get_spgemm_flop(sfCSR *a, sfCSR *b,
                     int M, long long int *flop)
{
    int GS, BS;
    long long int *d_max_row_nz;

    BS = MAX_LOCAL_THREAD_NUM;
    checkCudaErrors(cudaMalloc((void **)&(d_max_row_nz), sizeof(long long int) * M));
  
    GS = div_round_up(M, BS);
    set_intprod_per_row<<<GS, BS>>>(a->d_rpt, a->d_col,
                                    b->d_rpt,
                                    d_max_row_nz,
                                    M);
  
    long long int *tmp = (long long int *)malloc(sizeof(long long int) * M);
    cudaMemcpy(tmp, d_max_row_nz, sizeof(long long int) * M, cudaMemcpyDeviceToHost);
    *flop = thrust::reduce(thrust::device, d_max_row_nz, d_max_row_nz + M);

    (*flop) *= 2;
    cudaFree(d_max_row_nz);

}

void spgemm_kernel_cu_csr(sfCSR *a, sfCSR *b, sfCSR *c,
                          cusparseHandle_t *cusparseHandle,
                          cusparseOperation_t *trans_a,
                          cusparseOperation_t *trans_b,
                          cusparseMatDescr_t *descr_a,
                          cusparseMatDescr_t *descr_b)
{
    int m, n, k;
    int base_c, nnz_c;
    int *nnzTotalDevHostPtr = &nnz_c;
    cusparseMatDescr_t descr_c;
    cusparseStatus_t status;
  
    // int it = 0;
    // struct timeval start, end;
    // float msec[10];

    m = a->M;
    n = b->N;
    k = a->N;
    c->M = m;
    c->N = n;
  
    // gettimeofday(&start, NULL);
    cusparseCreateMatDescr(&descr_c);
    cusparseSetMatType(descr_c,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr_c,CUSPARSE_INDEX_BASE_ZERO);
    // gettimeofday(&end, NULL);
    // msec[it++] = (float)(end.tv_sec - start.tv_sec) * 1000 + (float)(end.tv_usec - start.tv_usec) / 1000;
  
    // msec[it++] = 0;

    // gettimeofday(&start, NULL);
    cusparseSetPointerMode(*cusparseHandle, CUSPARSE_POINTER_MODE_HOST);
    checkCudaErrors(cudaMalloc((void **)&(c->d_rpt), sizeof(int) * (m + 1)));
    // gettimeofday(&end, NULL);
    // msec[it++] = (float)(end.tv_sec - start.tv_sec) * 1000 + (float)(end.tv_usec - start.tv_usec) / 1000;

    /* Count nnz of C */
    // gettimeofday(&start, NULL);
    status = cusparseXcsrgemmNnz(*cusparseHandle, *trans_a, *trans_b, m, n, k,
                                 *descr_a, a->nnz, a->d_rpt, a->d_col,
                                 *descr_b, b->nnz, b->d_rpt, b->d_col,
                                 descr_c, c->d_rpt, nnzTotalDevHostPtr);
  
    if (status != CUSPARSE_STATUS_SUCCESS) {
        printf("Fail by xcsrgemmnnz\n");
        exit(1);
    }

    if (nnzTotalDevHostPtr != NULL) {
        c->nnz = *nnzTotalDevHostPtr;
    }
    else {
        cudaMemcpy(&(c->nnz), c->d_rpt + m, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&base_c, c->d_rpt, sizeof(int), cudaMemcpyDeviceToHost);
        c->nnz -= base_c;
    }
  
    // gettimeofday(&end, NULL);
    // msec[it++] = (float)(end.tv_sec - start.tv_sec) * 1000 + (float)(end.tv_usec - start.tv_usec) / 1000;

    // msec[it++] = 0;

    /* Calculating value of C */
    // gettimeofday(&start, NULL);

    checkCudaErrors(cudaMalloc((void **)&(c->d_col), sizeof(int) * c->nnz));
    checkCudaErrors(cudaMalloc((void **)&(c->d_val), sizeof(real) * c->nnz));

    // gettimeofday(&end, NULL);
    // msec[it++] = (float)(end.tv_sec - start.tv_sec) * 1000 + (float)(end.tv_usec - start.tv_usec) / 1000;

    // gettimeofday(&start, NULL);
#ifdef FLOAT
    status = cusparseScsrgemm(*cusparseHandle, *trans_a, *trans_b, m, n, k,
                              *descr_a, a->nnz,
                              a->d_val, a->d_rpt, a->d_col,
                              *descr_b, b->nnz,
                              b->d_val, b->d_rpt, b->d_col,
                              descr_c,
                              c->d_val, c->d_rpt, c->d_col);
#else
    status = cusparseDcsrgemm(*cusparseHandle, *trans_a, *trans_b, m, n, k,
                              *descr_a, a->nnz,
                              a->d_val, a->d_rpt, a->d_col,
                              *descr_b, b->nnz,
                              b->d_val, b->d_rpt, b->d_col,
                              descr_c,
                              c->d_val, c->d_rpt, c->d_col);
#endif
  
    cudaThreadSynchronize();
    if (status != CUSPARSE_STATUS_SUCCESS) {
        printf("Fail by csrgemm\n");
        exit(1);
    }
//     gettimeofday(&end, NULL);
//     msec[it++] = (float)(end.tv_sec - start.tv_sec) * 1000 + (float)(end.tv_usec - start.tv_usec) / 1000;

//     msec[it++] = 0;

// #ifdef EVAL_BD
//     int i;
//     if (eval_it > 0) {
//         printf("%s, cuSPARSE, %s", DATA_TYPE, a->matrix_name);
//         for (i = 0; i < it; i++) {
//             printf(", %f", msec[i]);
//         }
//         printf("\n");
//     }
//     eval_it++;
// #endif

}

void spgemm_cu_csr(sfCSR *a, sfCSR *b, sfCSR *c)
{
    cusparseHandle_t cusparseHandle;
    cusparseMatDescr_t descr_a, descr_b;
    cusparseOperation_t trans_a, trans_b;

    trans_a = trans_b = CUSPARSE_OPERATION_NON_TRANSPOSE;
  
    /* Set up cuSPARSE Library */
    cusparseCreate(&cusparseHandle);
    cusparseCreateMatDescr(&descr_a);
    cusparseCreateMatDescr(&descr_b);
    cusparseSetMatType(descr_a,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatType(descr_b,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr_a,CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatIndexBase(descr_b,CUSPARSE_INDEX_BASE_ZERO);
  
    /* Execution of SpMV on Device */
    spgemm_kernel_cu_csr(a, b, c,
                         &cusparseHandle,
                         &trans_a, &trans_b,
                         &descr_a, &descr_b);
    cudaThreadSynchronize();
    
    csr_memcpyDtH(c);

    release_csr(*c);
    cusparseDestroy(cusparseHandle);
}

