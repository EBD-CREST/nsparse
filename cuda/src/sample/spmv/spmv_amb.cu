#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include <time.h>

#include <math.h>

#include <cuda.h>
#include <helper_cuda.h>
#include <cuda_profiler_api.h>

#include <nsparse.h>

void spmv_amb(sfCSR *csr_mat, real *x, real *y, sfPlan *plan)
{
    int i;

    real *d_x, *d_y;
    sfAMB mat;

    cudaEvent_t event[2];
    struct timeval cstart, cend;
    float cmsec, msec, ave_msec, flops;
  
    for (i = 0; i < 2; i++) {
        cudaEventCreate(&(event[i]));
    }
  
    /* Malloc and memcpy HtoD */
    csr_memcpy(csr_mat);
    checkCudaErrors(cudaMalloc((void **)&d_x, sizeof(real) * (csr_mat->N + MAX_BLOCK_SIZE)));
    checkCudaErrors(cudaMalloc((void **)&d_y, sizeof(real) * (csr_mat->M + WARP)));
    checkCudaErrors(cudaMemcpy(d_x, x, sizeof(real) * csr_mat->N, cudaMemcpyHostToDevice));
  
    /* Converting format from CSR to AMB */
    gettimeofday(&cstart, NULL);
    sf_csr2amb(&mat, csr_mat, d_x, plan);
    gettimeofday(&cend, NULL);
  
    cmsec = (float)(cend.tv_sec - cstart.tv_sec) * 1000 + (float)(cend.tv_usec - cstart.tv_usec) / 1000;
    printf("Format Conversion Cost (CSR=>AMB, %d-%d): %f[msec]\n", mat.seg_size, mat.block_size, cmsec);
  
    /* Execution of SpMV on Device */
    ave_msec = 0;
    for (i = 0; i < TRI_NUM; i++) {
        cudaEventRecord(event[0], 0);
        sf_spmv_amb(d_y, &mat, d_x, plan);
        cudaEventRecord(event[1], 0);
        cudaThreadSynchronize();
	
        cudaEventElapsedTime(&msec, event[0], event[1]);
        if (i > 0) {
            ave_msec += msec;
        }
    }

    ave_msec /= TRI_NUM - 1;
  
    checkCudaErrors(cudaMemcpy(y, d_y, sizeof(real) * csr_mat->M, cudaMemcpyDeviceToHost));
  
    flops = (real)(csr_mat->nnz) * 2 / 1000 / 1000 / ave_msec;
  
    printf("SpMV using AMB format: %s, %f[GFLOPS], %f[ms]\n", csr_mat->matrix_name, flops, ave_msec);

    /*7th step. Release*/
    cudaFree(d_x);
    cudaFree(d_y);
    release_amb(mat);
    release_csr(*csr_mat);
  
}

/*Main Function*/
int main(int argc, char **argv)
{
    sfCSR mat;
    real *x, *y;
    sfPlan plan;
  
    /* Set CSR reading from MM file or generating random matrix */
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

    /* Format conversion */
    if (argc >= 3) { //Manual Parameter Setting
        set_plan(&plan, atoi(argv[2]), atoi(argv[3]));
    }
    else { //Auto-tuning Parameter Setting
        init_plan(&plan);
    }
  
    /* SpMV on GPU */
    spmv_amb(&mat, x, y, &plan);

    /* Check answer */
#ifdef sfDEBUG
    ans_check(csr_y, y, mat.M);
    free(csr_y);
#endif

    free(x);
    free(y);
    release_cpu_csr(mat);
  
    return 0;
}
