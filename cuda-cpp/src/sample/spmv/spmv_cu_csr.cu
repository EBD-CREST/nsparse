#include <iostream>
#include <cfloat>

#include <cuda.h>
#include <helper_cuda.h>
#include <cusparse_v2.h>

#include <CSR.hpp>
#include <nsparse.hpp>

typedef int IT;
#ifdef FLOAT
typedef float VT;
#else
typedef double VT;
#endif

template <class idType, class valType>
void spmv_cu_csr(CSR<idType, valType> &mat, const valType *x, valType *y)
{
    idType i;
    valType *d_x, *d_y;

    cudaEvent_t event[2];
    float exe_msec, ave_msec, flops;

    const valType alpha = 1.0;
    const valType beta = 0.0;
    cusparseHandle_t cusparseHandle = 0;
    cusparseMatDescr_t descr = 0;

    for (i = 0; i < 2; i++) {
        cudaEventCreate(&(event[i]));
    }
  
    /* Malloc and memcpy HtoD */
    mat.memcpyHtD();
  
    checkCudaErrors(cudaMalloc((void **)&d_x, sizeof(valType) * mat.ncolumn));
    checkCudaErrors(cudaMalloc((void **)&d_y, sizeof(valType) * mat.nrow));
    checkCudaErrors(cudaMemcpy(d_x, x, sizeof(valType) * mat.ncolumn, cudaMemcpyHostToDevice));

    /* Set up of cuSPARSE */
    cusparseCreate(&cusparseHandle);
    cusparseCreateMatDescr(&descr);
    cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);
  
    /* Execution of SpMV on Device */
    ave_msec = 0;
    for (i = 0; i < TRI_NUM; i++) {
        cudaEventRecord(event[0], 0);
#ifdef FLOAT
        cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                       mat.nrow, mat.ncolumn, mat.nnz,
                       &alpha, descr, mat.d_values, mat.d_rpt, mat.d_colids,
                       d_x, &beta, d_y);
#else
        cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                       mat.nrow, mat.ncolumn, mat.nnz,
                       &alpha, descr, mat.d_values, mat.d_rpt, mat.d_colids,
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
  
    checkCudaErrors(cudaMemcpy(y, d_y, sizeof(valType) * mat.nrow, cudaMemcpyDeviceToHost));

    flops = (float)(mat.nnz) * 2 / 1000 / 1000 / ave_msec;
    printf("SpMV using CSR format (cuSPARSE): %f[GFLOPS], %f[ms]\n", flops, ave_msec);

    /* Release memory object*/
    cudaFree(d_x);
    cudaFree(d_y);
    mat.release_csr();
    cusparseDestroy(cusparseHandle);

}

/*Main Function*/
int main(int argc, char *argv[])
{
    CSR<IT, VT> mat;
    VT *x, *y;

    /* Set CSR reding from MM file or generating random matrix */
    cout << "Read matrix data from " << argv[1] << endl;
    mat.init_data_from_mtx(argv[1]);
  
    /* Init vectors on CPU */
    x = new VT[mat.ncolumn];
    y = new VT[mat.nrow];
    
    init_vector<IT, VT>(x, mat.ncolumn);
    
    /* Execution of SpMV on GPU */
    spmv_cu_csr<IT, VT>(mat, x, y);

#ifdef sfDEBUG
    /* Execution of SpMV on CPU */
    VT *ans_y = new VT[mat.nrow];
    mat.spmv_cpu(x, ans_y);
    check_answer<IT, VT>(ans_y, y, mat.nrow);
    delete[] ans_y;
#endif

    delete[] x;
    delete[] y;
    mat.release_cpu_csr();
  
    return 0;

}

