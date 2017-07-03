#include <stdio.h>
#include <stdlib.h>
#include <nsparse.h>

#define MAX_BIN_NUM 100
#define TB_NUM_WR 4

typedef enum {
    GTR,
    TR,
    PWR
} PATTERN;
  
int main(int argc, char **argv)
{
  
    int i;
    int count;
    int shared_tb, shared_sm;
    int tb_max, tb_sm;
    int max_tb_num_sm;
    int pw_min_table, min_table;
  
    int bin_num;
  
    PATTERN pat[MAX_BIN_NUM];
    int table_size[MAX_BIN_NUM], tb[MAX_BIN_NUM];
    int tb_tr;
    int pwarp;
  
    pw_min_table = 16;

    printf("For value data with %d bytes\n", sizeof(real));

    /*
     * Architecture specific parameters
     * You set appropriate values with your target GPU
     */
    /************************/
    shared_tb = 48 * 1024; // Maximum Shared Memory Size / Thread Block
    shared_sm = 64 * 1024; // Maximum Shared Memory Size / SM
    tb_max = 1024; // Max Thread Block Size
    tb_sm = 2048; // Max Threads / Multiprocessor
    max_tb_num_sm = 32; // Max Thread Blocks / Multiprocessor
    /************************/

    pwarp = 4;
    bin_num = 1;
  
    /* Set largest hash table size */
    table_size[0] = shared_tb / (sizeof(int) + sizeof(real));
  
    count = 0;
    while (table_size[0] != 0) {
        table_size[0] >>= 1;
        count++;
    }
    table_size[0] = 1 << (count - 1);
    tb[0] = tb_max;
    pat[0] = GTR;
  
    table_size[1] = table_size[0];
    tb[1] = tb[0];
    pat[1] = TR;

    /* Set table size and thread block size for TB/ROW */
    i = 1;
    while (1) {
        i++;
        table_size[i] = table_size[i - 1] / 2;
        tb_tr = tb[i - 1] / 2;

        count = 0;
        if (tb_sm / tb_tr > max_tb_num_sm) {
            break;
        }
        tb[i] = tb_tr;
        pat[i] = TR;
    }
    bin_num = i + 1;

    /* Set table size and thread block size for PWARP/ROW */
    table_size[i] = pw_min_table;
    tb[i] = (shared_sm * pwarp) / ((sizeof(int) + sizeof(real)) * pw_min_table * TB_NUM_WR);
    count = 0;
    while (tb[i] != 0) {
        tb[i] >>= 1;
        count++;
    }
    tb[i] = 1 << (count - 1);
    pat[i] = PWR;
  
    bin_num = i + 1;
  
    min_table = table_size[bin_num - 2];

    printf("bin_num : %d\n", bin_num);
    printf("Hash table size: ");
    for (i = 0; i < bin_num; i++) {
        printf("%d, ", table_size[i]);
    }
    printf("\n");

    printf("Thread block size: ");
    for (i = 0; i < bin_num; i++) {
        printf("%d, ", tb[i]);
    }
    printf("\n");

    printf("Thread Assignment: ");
    for (i = 0; i < bin_num; i++) {
        switch (pat[i]) {
        case GTR:
            printf("GTR, ");
            break;
        case TR:
            printf("TR, ");
            break;
        case PWR:
            printf("PWR, ");
            break;
        default:
            printf("NO, ");
        }
    }
    printf("\n");

    /* Generate kernel file */
    FILE *fp;
#ifdef FLOAT
    char kernel_name[100] = "./src/kernel/kernel_spgemm_hash_s.cu";
#else
    char kernel_name[100] = "./src/kernel/kernel_spgemm_hash_d.cu";
#endif
  
    /* Define macro */
    fp = fopen(kernel_name, "w");
    fprintf(fp, "#define BIN_NUM %d\n", bin_num);
    fprintf(fp, "#define PWARP %d\n", pwarp);
    fprintf(fp, "#define IMB_PWMIN %d\n", pw_min_table * 2);
    fprintf(fp, "#define B_PWMIN %d\n", pw_min_table);
    fprintf(fp, "#define IMB_MIN %d\n", min_table * 2);
    fprintf(fp, "#define B_MIN %d\n", min_table);
    fprintf(fp, "#define IMB_PW_SH_SIZE %d\n", 2 * pw_min_table * tb[bin_num - 1] / pwarp);
    fprintf(fp, "#define B_PW_SH_SIZE %d\n", pw_min_table * tb[bin_num - 1] / pwarp);
    fprintf(fp, "#define IMB_SH_SIZE %d\n", (2 * table_size[bin_num - 2]) * (tb[bin_num - 2] / WARP));
    fprintf(fp, "#define B_SH_SIZE %d\n", (table_size[bin_num - 2]) * (tb[bin_num - 2] / WARP));
    fprintf(fp, "\n");
    fclose(fp);

    /* Copy */
#ifdef FLOAT
    system("cat ./src/kernel/kernel_spgemm_hash_template.cu >> ./src/kernel/kernel_spgemm_hash_s.cu");
#else
    system("cat ./src/kernel/kernel_spgemm_hash_template.cu >> ./src/kernel/kernel_spgemm_hash_d.cu");
#endif

    /* Write kernel part */
    fp = fopen(kernel_name, "a");
    /* Count nnz kernel */
    fprintf(fp, "void set_row_nnz(int *d_arpt, int *d_acol,\n\
                 int *d_brpt, int *d_bcol,\n\
                 int *d_crpt,\n                     \
                 sfBIN *bin,\n\
                 int M,\n\
                 int *nnz)\n\
{\n                                            \
    int i;\n\
    int GS, BS;\n\
    for (i = BIN_NUM - 1; i >= 0; i--) {\n\
        if (bin->bin_size[i] > 0) {\n\
            switch (i) {\n\
            case 0:\n\
                BS = %d;\n\
                GS = div_round_up(bin->bin_size[i] * PWARP, BS);\n\
                set_row_nz_bin_pwarp<<<GS, BS, 0, bin->stream[i]>>>\n\
                    (d_arpt, d_acol,\n\
                    d_brpt, d_bcol,\n                     \
                    bin->d_row_perm,\n\
                    bin->d_row_nz,\n\
                    bin->bin_offset[i],\n\
                    bin->bin_size[i]);\n\
                break;\n\
", tb[bin_num - 1]);
    for (i = 1; i < bin_num - 1; i++) {
        fprintf(fp, "            case %d :\n                 \
            	BS = %d;\n				\
            	GS = bin->bin_size[i];\n\
            	set_row_nz_bin_each_tb<%d><<<GS, BS, 0, bin->stream[i]>>>\n\
            	  (d_arpt, d_acol, d_brpt, d_bcol,\n\
            	   bin->d_row_perm, bin->d_row_nz,\n\
            	   bin->bin_offset[i], bin->bin_size[i]);\n\
            	break;\n\
", i, tb[bin_num - 1 - i], table_size[bin_num - 1 - i] * 2);
    }

    fprintf(fp, "            case %d :\n                         \
            	{\n\
            	    int fail_count;\n\
            	    int *d_fail_count, *d_fail_perm;\n\
            	    fail_count = 0;\n\
            	    checkCudaErrors(cudaMalloc((void **)&d_fail_count, sizeof(int)));\n\
            	    checkCudaErrors(cudaMalloc((void **)&d_fail_perm, sizeof(int) * bin->bin_size[i]));\n\
            	    cudaMemcpy(d_fail_count, &fail_count, sizeof(int), cudaMemcpyHostToDevice);\n\
            	    BS = %d;\n\
            	    GS = bin->bin_size[i];\n\
            	    set_row_nz_bin_each_tb_large<%d><<<GS, BS, 0, bin->stream[i]>>>\n\
            	      (d_arpt, d_acol, d_brpt, d_bcol,\n\
            	       bin->d_row_perm, bin->d_row_nz,\n\
            	       d_fail_count, d_fail_perm,\n\
            	       bin->bin_offset[i], bin->bin_size[i]);\n\
            	    cudaMemcpy(&fail_count, d_fail_count, sizeof(int), cudaMemcpyDeviceToHost);\n\
            	    if (fail_count > 0) {\n\
              	        int max_row_nz = bin->max_intprod;\n\
            	        size_t table_size = (size_t)max_row_nz * fail_count;\n\
            	        int *d_check;\n\
            	        checkCudaErrors(cudaMalloc((void **)&(d_check), sizeof(int) * table_size));\n\
            	        BS = %d;\n\
            	        GS = div_round_up(table_size, BS);\n\
            	        init_check<<<GS, BS, 0, bin->stream[i]>>>(d_check, table_size);\n\
            	        GS = bin->bin_size[i];\n\
	                    set_row_nz_bin_each_gl<<<GS, BS, 0, bin->stream[i]>>>\n   \
                  		  (d_arpt, d_acol, d_brpt, d_bcol,\n\
		                   d_fail_perm, bin->d_row_nz, d_check,\n             \
		                   max_row_nz, 0, fail_count);\n                                  \
	                    cudaFree(d_check);\n  \
	                }\n\
	            cudaFree(d_fail_count);\n\
	            cudaFree(d_fail_perm);\n\
	        }\n\
	        break;\n\
	      default :\n\
	          exit(0);\n\
	      }\n\
        }\n\
      }\n\
      cudaThreadSynchronize();\n\
\n\
    /* Set row pointer of matrix C */\n\
    thrust::exclusive_scan(thrust::device, bin->d_row_nz, bin->d_row_nz + (M + 1), d_crpt, 0);\n\
    cudaMemcpy(nnz, d_crpt + M, sizeof(int), cudaMemcpyDeviceToHost);\n\
}\
\n\n", bin_num - 1, tb[0], table_size[0] * 2, tb[0]);
  
    /* Calculation kernel */
    fprintf(fp, "void calculate_value_col_bin(int *d_arpt, int *d_acol, real *d_aval,\n\
			     int *d_brpt, int *d_bcol, real *d_bval,\n\
			     int *d_crpt, int *d_ccol, real *d_cval,\n\
			     sfBIN *bin,\n\
			     int M)\n\
{\n                     \
  int i;\n\
  int GS, BS;\n\
  for (i = BIN_NUM - 1; i >= 0; i--) {\n\
    if (bin->bin_size[i] > 0) {\n\
      switch (i) {\n\
      case 0:\n\
      BS = %d;\n\
      GS = div_round_up(bin->bin_size[i] * PWARP, BS);\n\
      calculate_value_col_bin_pwarp<<<GS, BS, 0, bin->stream[i]>>>\n\
           (d_arpt, d_acol, d_aval,\n\
	   d_brpt, d_bcol, d_bval,\n\
	   d_crpt, d_ccol, d_cval,\n\
	   bin->d_row_perm, bin->d_row_nz,\n\
	   bin->bin_offset[i], bin->bin_size[i]);\n\
      break;\n\
", tb[bin_num - 1]);
    for (i = 1; i < bin_num - 1; i++) {
        fprintf(fp, "      case %d:\n\
	  BS = %d;\n\
	  GS = bin->bin_size[i];\n\
	  calculate_value_col_bin_each_tb<%d><<<GS, BS, 0, bin->stream[i]>>>\n\
	    (d_arpt, d_acol, d_aval,\n\
	     d_brpt, d_bcol, d_bval,\n\
	     d_crpt, d_ccol, d_cval,\n\
	     bin->d_row_perm, bin->d_row_nz,\n\
	     bin->bin_offset[i], bin->bin_size[i]);\n\
	  break;\n\
", i, tb[bin_num - 1 - i], table_size[bin_num - 1 - i]);
    }
    fprintf(fp, "	case %d :\n\
	  {\n\
	    int max_row_nz = bin->max_nz * 2;\n\
	    int table_size = max_row_nz * bin->bin_size[i];\n\
	    int *d_check;\n\
	    real *d_value;\n\
	    checkCudaErrors(cudaMalloc((void **)&(d_check), sizeof(int) * table_size));\n\
	    checkCudaErrors(cudaMalloc((void **)&(d_value), sizeof(real) * table_size));\n\
	    BS = %d;\n\
	    GS = div_round_up(table_size, BS);\n\
	    init_check<<<GS, BS, 0, bin->stream[i]>>>(d_check, table_size);\n\
	    init_value<<<GS, BS, 0, bin->stream[i]>>>(d_value, table_size);\n\
	    GS = bin->bin_size[i];\n\
	    calculate_value_col_bin_each_gl<<<GS, BS, 0, bin->stream[i]>>>\n\
	      (d_arpt, d_acol, d_aval,\n\
	       d_brpt, d_bcol, d_bval,\n\
	       d_crpt, d_ccol, d_cval,\n\
	       bin->d_row_perm, bin->d_row_nz,\n\
	       d_check, d_value, max_row_nz,\n\
	       bin->bin_offset[i], bin->bin_size[i]);\n\
	    cudaFree(d_check);\n\
	    cudaFree(d_value);\n\
	  }\n\
	  break;\n\
	default :\n\
	  exit(0);\n\
	}\n\
      }\n\
    }\n\
  cudaThreadSynchronize();\n\
}\n\
", bin_num - 1, tb[0]);
    fclose(fp);


    return 0;

}
