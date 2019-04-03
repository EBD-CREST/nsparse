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
#include <HashSpGEMM_volta.hpp>

typedef int IT;
#ifdef FLOAT
typedef float VT;
#else
typedef double VT;
#endif

template <class idType, class valType>
void spgemm_hash(CSR<idType, valType> a, CSR<idType, valType> b, CSR<idType, valType> &c)
{

    idType i;
  
    long long int flop_count;
    cudaEvent_t event[2];
    float msec, ave_msec, flops;
  
    for (i = 0; i < 2; i++) {
        cudaEventCreate(&(event[i]));
    }
  
    /* Memcpy A and B from Host to Device */
    a.memcpyHtD();
    b.memcpyHtD();
  
    /* Count flop of SpGEMM computation */
    get_spgemm_flop(a, b, flop_count);

    /* Execution of SpGEMM on Device */
    ave_msec = 0;
    for (i = 0; i < SpGEMM_TRI_NUM; i++) {
        if (i > 0) {
            c.release_csr();
        }
        cudaEventRecord(event[0], 0);
        SpGEMM_Hash(a, b, c);
        cudaEventRecord(event[1], 0);
        cudaDeviceSynchronize();
        cudaEventElapsedTime(&msec, event[0], event[1]);
    
        if (i > 0) {
            ave_msec += msec;
        }
    }
    ave_msec /= SpGEMM_TRI_NUM - 1;

    flops = (float)(flop_count) / 1000 / 1000 / ave_msec;
    printf("SpGEMM using CSR format (Hash): %f[GFLOPS], %f[ms]\n", flops, ave_msec);

    /* Numeric Only */
    ave_msec = 0;
    for (i = 0; i < SpGEMM_TRI_NUM; i++) {
        cudaEventRecord(event[0], 0);
        SpGEMM_Hash_Numeric(a, b, c);
        cudaEventRecord(event[1], 0);
        cudaDeviceSynchronize();
        cudaEventElapsedTime(&msec, event[0], event[1]);
    
        if (i > 0) {
            ave_msec += msec;
        }
    }
    ave_msec /= SpGEMM_TRI_NUM - 1;

    flops = (float)(flop_count) / 1000 / 1000 / ave_msec;
    printf("SpGEMM using CSR format (Hash, only numeric phase): %f[GFLOPS], %f[ms]\n", flops, ave_msec);

    c.memcpyDtH();
    c.release_csr();

#ifdef sfDEBUG
    CSR<IT, VT> cusparse_c;
    SpGEMM_cuSPARSE(a, b, cusparse_c);
    if (c == cusparse_c) {
        cout << "HashSpGEMM is correctly executed" << endl;
    }
    cout << "Nnz of A: " << a.nnz << endl; 
    cout << "Number of intermediate products: " << flop_count / 2 << endl; 
    cout << "Nnz of C: " << c.nnz << endl; 
    cusparse_c.release_cpu_csr();
#endif

    a.release_csr();
    b.release_csr();

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
    spgemm_hash(a, b, c);
    
    a.release_cpu_csr();
    b.release_cpu_csr();
    c.release_cpu_csr();
  
    return 0;

}

