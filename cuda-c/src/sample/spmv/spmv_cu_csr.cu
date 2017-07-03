#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <math.h>

#include <cuda.h>
#include <helper_cuda.h>
#include <cusparse_v2.h>

#include <nsparse.h>

void spmv_cu_csr(sfCSR *mat, real *x, real *y)
{
    int i;
    real *d_x, *d_y;

    cudaEvent_t event[2];
    float exe_msec, min_msec, ave_msec, flops;

    const real alpha = 1.0;
    const real beta = 0.0;
    cusparseHandle_t cusparseHandle = 0;
    cusparseMatDescr_t descr = 0;

    for (i = 0; i < 2; i++) {
        cudaEventCreate(&(event[i]));
    }
  
    /* Malloc and memcpy HtoD */
    csr_memcpy(mat);
  
    checkCudaErrors(cudaMalloc((void **)&d_x, sizeof(real) * mat->N));
    checkCudaErrors(cudaMalloc((void **)&d_y, sizeof(real) * mat->M));
  
    checkCudaErrors(cudaMemcpy(d_x, x, sizeof(real) * mat->N, cudaMemcpyHostToDevice));

    /* Set up of cuSPARSE */
    cusparseCreate(&cusparseHandle);
    cusparseCreateMatDescr(&descr);
    cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);
  
    /* Execution of SpMV on Device */
    min_msec = sfFLT_MAX;
    ave_msec = 0;
    for (i = 0; i < TRI_NUM; i++) {
        cudaEventRecord(event[0], 0);
#ifdef FLOAT
        cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                       mat->M, mat->N, mat->nnz,
                       &alpha, descr, mat->d_val, mat->d_rpt, mat->d_col,
                       d_x, &beta, d_y);
#else
        cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                       mat->M, mat->N, mat->nnz,
                       &alpha, descr, mat->d_val, mat->d_rpt, mat->d_col,
                       d_x, &beta, d_y);
#endif
        cudaEventRecord(event[1], 0);
    
        cudaThreadSynchronize();
    
        cudaEventElapsedTime(&exe_msec, event[0], event[1]);
        if (i > 0) {
            ave_msec += exe_msec;
        }
    }
    ave_msec /= TRI_NUM - 1;
    if (min_msec > ave_msec) {
        min_msec = ave_msec;
    }
  
    checkCudaErrors(cudaMemcpy(y, d_y, sizeof(real) * mat->M, cudaMemcpyDeviceToHost));

    flops = (float)(mat->nnz) * 2 / 1000 / 1000 / min_msec;
    printf("SpMV using CSR format (cuSPARSE): %s, %f[GFLOPS], %f[ms]\n", mat->matrix_name, flops, ave_msec);

    /* Release memory object*/
    cudaFree(d_x);
    cudaFree(d_y);
    release_csr(*mat);
    cusparseDestroy(cusparseHandle);

}


/*Main Function*/
int main(int argc, char *argv[]) {

    sfCSR mat;
    real *x, *y;

    /* Set CSR reding from MM file or generating random matrix */
    init_csr_matrix_from_file(&mat, argv[1]);
  
    /* Init vectors on CPU */
    x = (real *)malloc(sizeof(real) * mat.N);
    y = (real *)malloc(sizeof(real) * mat.M);
    init_vector(x, mat.N);

    /* Execution of SpMV on CPU */
#ifdef sfDEBUG
    real *csr_y;
    csr_y = (real *)malloc(sizeof(real) * mat.M);
    csr_kernel(csr_y, &mat, x);
#endif

    spmv_cu_csr(&mat, x, y);

#ifdef sfDEBUG
    ans_check(csr_y, y, mat.M);
    free(csr_y);
#endif

    free(x);
    free(y);
    release_cpu_csr(mat);
  
    return 0;

}

