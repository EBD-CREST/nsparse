#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <helper_cuda.h>
#include <cusparse_v2.h>

#include <nsparse.h>

void sf_spmv_cu_csr(real *d_y, sfCSR *mat, real *d_x,
                    cusparseHandle_t *cusparseHandle,
                    cusparseMatDescr_t *descr) {
  
    const real alpha = 1.0;
    const real beta = 0.0;

#ifdef FLOAT
    cusparseScsrmv(*cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE,
                   mat->M, mat->N, mat->nnz,
                   &alpha, *descr, mat->d_val, mat->d_rpt, mat->d_col,
                   d_x, &beta, d_y);
#else
    cusparseDcsrmv(*cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE,
                   mat->M, mat->N, mat->nnz,
                   &alpha, *descr, mat->d_val, mat->d_rpt, mat->d_col,
                   d_x, &beta, d_y);
#endif

    cudaThreadSynchronize();
      
}

