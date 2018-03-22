#include <iostream>
#include <cfloat>

#include <helper_cuda.h>
#include <cusparse_v2.h>

#include <nsparse.hpp>
#include <CSR.hpp>
#include <AMB.hpp>

typedef int IT;
typedef unsigned short compIT;
#ifdef FLOAT
typedef float VT;
#else
typedef double VT;
#endif

template <class idType, class valType>
void spmv(CSR<idType, valType> &mat, const valType *x, valType *y, Plan<idType> &plan)
{
    idType i;
    valType *d_x, *d_y;
    AMB<idType, compIT, valType> a_mat;

    cudaEvent_t event[2];
    float exe_msec, min_msec, ave_msec, flops;

    for (i = 0; i < 2; i++) {
        cudaEventCreate(&(event[i]));
    }
  
    /* Malloc and memcpy HtoD */
    mat.memcpyHtD();

    checkCudaErrors(cudaMalloc((void **)&d_x, sizeof(valType) * mat.ncolumn));
    checkCudaErrors(cudaMalloc((void **)&d_y, sizeof(valType) * mat.nrow));
    checkCudaErrors(cudaMemcpy(d_x, x, sizeof(valType) * mat.ncolumn, cudaMemcpyHostToDevice));

    /* Converting format from CSR to AMB */
    a_mat.convert_from_csr(mat, plan, d_x);
    printf("Format Conversion Cost (CSR=>AMB, %d-%d)\n", a_mat.seg_size, a_mat.block_size);
    cout << "Format Conversion: CSR => AMB(" << a_mat.seg_size << ", " << a_mat.block_size << ", " << plan.thread_block << ", " << plan.thread_grid << ")" << endl;

    /* Execution of SpMV on Device */
    min_msec = FLT_MAX;
    ave_msec = 0;
    for (i = 0; i < TRI_NUM; i++) {
        cudaEventRecord(event[0], 0);
        a_mat.spmv(d_x, d_y, plan);
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
  
    checkCudaErrors(cudaMemcpy(y, d_y, sizeof(valType) * mat.nrow, cudaMemcpyDeviceToHost));

    flops = (float)(mat.nnz) * 2 / 1000 / 1000 / min_msec;
    printf("SpMV using AMB format : %f[GFLOPS], %f[ms]\n", flops, ave_msec);

    /* Release memory object*/
    cudaFree(d_x);
    cudaFree(d_y);
    a_mat.release_amb();
    mat.release_csr();

}

/*Main Function*/
int main(int argc, char *argv[])
{
    CSR<IT, VT> mat;
    VT *x, *y;
    Plan<IT> plan;

    /* Set CSR reding from MM file or generating random matrix */
    cout << "Read matrix data from " << argv[1] << endl;
    mat.init_data_from_mtx(argv[1]);
  
    /* Init vectors on CPU */
    x = new VT[mat.ncolumn];
    y = new VT[mat.nrow];
    
    init_vector<IT, VT>(x, mat.ncolumn);
    
    /* Parameter set */
    if (argc >= 3) {
        plan.set_plan(atoi(argv[2]), atoi(argv[3]));
    }
    
    /* Execution of SpMV on GPU */
    spmv<IT, VT>(mat, x, y, plan);
    
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

