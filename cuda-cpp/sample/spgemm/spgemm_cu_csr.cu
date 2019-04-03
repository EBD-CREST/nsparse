#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <math.h>

#include <cuda.h>
#include <helper_cuda.h>
#include <cusparse_v2.h>

#include <nsparse.hpp>
#include <CSR.hpp>
#include <SpGEMM.hpp>

typedef int IT;
#ifdef FLOAT
typedef float VT;
#else
typedef double VT;
#endif

template <class idType, class valType>
void spgemm_cu_csr(CSR<idType, valType> a, CSR<idType, valType> b, CSR<idType, valType> &c)
{

    idType i;
  
    long long int flop_count;
    cudaEvent_t event[2];
    float msec, ave_msec, flops;
  
    cusparseHandle_t cusparseHandle;
    cusparseMatDescr_t descr_a, descr_b, descr_c;
    cusparseOperation_t trans_a, trans_b;

    for (i = 0; i < 2; i++) {
        cudaEventCreate(&(event[i]));
    }
    trans_a = trans_b = CUSPARSE_OPERATION_NON_TRANSPOSE;
  
    /* Memcpy A and B from Host to Device */
    a.memcpyHtD();
    b.memcpyHtD();
  
    /* Count flop of SpGEMM computation */
    get_spgemm_flop(a, b, flop_count);

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
  
    /* Execution of SpGEMM on Device */
    ave_msec = 0;
    for (i = 0; i < SpGEMM_TRI_NUM; i++) {
        if (i > 0) {
            c.release_csr();
        }
        cudaEventRecord(event[0], 0);
        SpGEMM_cuSPARSE_kernel(a, b, c, cusparseHandle, trans_a, trans_b, descr_a, descr_b, descr_c);
        cudaEventRecord(event[1], 0);
        cudaDeviceSynchronize();
        cudaEventElapsedTime(&msec, event[0], event[1]);
    
        if (i > 0) {
            ave_msec += msec;
        }
    }
    ave_msec /= SpGEMM_TRI_NUM - 1;

    flops = (float)(flop_count) / 1000 / 1000 / ave_msec;
    printf("SpGEMM using CSR format (cuSPARSE): %f[GFLOPS], %f[ms]\n", flops, ave_msec);

    c.memcpyDtH();

    a.release_csr();
    b.release_csr();
    c.release_csr();

    cusparseDestroy(cusparseHandle);
    for (i = 0; i < 2; i++) {
        cudaEventDestroy(event[i]);
    }
}

/*Main Function*/
int main(int argc, char *argv[])
{
    CSR<IT, VT> a, b, c;

    /* Set CSR reding from MM file or generating random matrix */
    cout << "Initialize Matrix A" << endl;
    cout << "Read matrix data from " << argv[1] << endl;
    a.init_data_from_mtx(argv[1]);

    cout << "Initialize Matrix B" << endl;
    cout << "Read matrix data from " << argv[1] << endl;
    b.init_data_from_mtx(argv[1]);
  
    /* Execution of SpGEMM on GPU */
    spgemm_cu_csr(a, b, c);
    
    a.release_cpu_csr();
    b.release_cpu_csr();
    c.release_cpu_csr();
  
    return 0;

}

