#include <iostream>
#include <math.h>

#include <cuda.h>
#include <helper_cuda.h>

#include <CSR.hpp>
#include <nsparse.hpp>

#define LINE_LENGTH_MAX 256

// void release_cpu_amb(sfAMB mat)
// {
//     free(mat.cs);
//     free(mat.cl);
//     free(mat.sellcs_val);
//     free(mat.sellcs_col);
//     free(mat.s_write_permutation);
//     free(mat.s_write_permutation_offset);
// }

// void release_amb(sfAMB mat)
// {
//     cudaFree(mat.d_cs);
//     cudaFree(mat.d_cl);
//     cudaFree(mat.d_sellcs_val);
//     cudaFree(mat.d_sellcs_col);
//     cudaFree(mat.d_write_permutation);
//     cudaFree(mat.d_s_write_permutation);
//     cudaFree(mat.d_s_write_permutation_offset);
// }

