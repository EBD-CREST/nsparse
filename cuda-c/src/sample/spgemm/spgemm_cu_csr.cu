#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <math.h>

#include <cuda.h>
#include <helper_cuda.h>
#include <cusparse_v2.h>

#include <nsparse.h>

void spgemm_csr(sfCSR *a, sfCSR *b, sfCSR *c)
{

    int i;
  
    long long int flop_count;
    cudaEvent_t event[2];
    float msec, ave_msec, flops;
  
    cusparseHandle_t cusparseHandle;
    cusparseMatDescr_t descr_a, descr_b;
    cusparseOperation_t trans_a, trans_b;

    for (i = 0; i < 2; i++) {
        cudaEventCreate(&(event[i]));
    }
    trans_a = trans_b = CUSPARSE_OPERATION_NON_TRANSPOSE;
  
    /* Memcpy A and B from Host to Device */
    csr_memcpy(a);
    csr_memcpy(b);
  
    /* Count flop of SpGEMM computation */
    get_spgemm_flop(a, b, a->M, &flop_count);

    /* Set up cuSPARSE Library */
    cusparseCreate(&cusparseHandle);
    cusparseCreateMatDescr(&descr_a);
    cusparseCreateMatDescr(&descr_b);
    cusparseSetMatType(descr_a,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatType(descr_b,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr_a,CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatIndexBase(descr_b,CUSPARSE_INDEX_BASE_ZERO);
  
    /* Execution of SpGEMM on Device */
    ave_msec = 0;
    for (i = 0; i < SPGEMM_TRI_NUM; i++) {
        if (i > 0) {
            release_csr(*c);
        }
        cudaEventRecord(event[0], 0);
        spgemm_kernel_cu_csr(a, b, c,
                             &cusparseHandle,
                             &trans_a, &trans_b,
                             &descr_a, &descr_b);
        cudaEventRecord(event[1], 0);
        cudaThreadSynchronize();
        cudaEventElapsedTime(&msec, event[0], event[1]);
    
        if (i > 0) {
            ave_msec += msec;
        }
    }
    ave_msec /= SPGEMM_TRI_NUM - 1;

    flops = (float)(flop_count) / 1000 / 1000 / ave_msec;
    printf("SpGEMM using CSR format (cuSPARSE): %s, %f[GFLOPS], %f[ms]\n", a->matrix_name, flops, ave_msec);

#ifdef sfDEBUG
    printf("(nnz of A): %d =>\n(Num of intermediate products): %ld =>\n(nnz of C): %d\n", a->nnz, flop_count / 2, c->nnz);
#endif

    release_csr(*a);
    release_csr(*b);
    release_csr(*c);
    cusparseDestroy(cusparseHandle);
    for (i = 0; i < 2; i++) {
        cudaEventDestroy(event[i]);
    }
}

/*Main Function*/
int main(int argc, char **argv)
{
    sfCSR mat_a, mat_b, mat_c;
    
    /* Set CSR reding from MM file or generating random matrix */
    init_csr_matrix_from_file(&mat_a, argv[1]);
    init_csr_matrix_from_file(&mat_b, argv[1]);

    spgemm_csr(&mat_a, &mat_b, &mat_c);

    release_cpu_csr(mat_a);
    release_cpu_csr(mat_b);
  
    return 0;

}
