#include <stdio.h>
#include <stdlib.h>

#include <math.h>

#include <cuda.h>
#include <helper_cuda.h>

#include <nsparse.h>

#define LINE_LENGTH_MAX 256

/*Convert file to CSR*/
void convert_file_csr(char *file_name,
                      int **rpt, int **col, real **val,
                      int *M, int *N, int *nz, int *nnz_max)
{
    int i;
    int isUnsy;
    int num, offset;
    char *line, *ch;
    FILE *fp;
    int *col_coo, *row_coo;
    real *val_coo;
    int *each_row_index;

    int *nnz_num, *col_, *rpt_;
    real *val_;

    isUnsy = 0;
    line = (char *)malloc(sizeof(char) * LINE_LENGTH_MAX);
  
    /* Open File */
    fp = fopen(file_name, "r");
    if(fp == NULL) {
        printf("Cannot find file\n");
        exit(1);
    }
    printf("Read mtx file: %s\n", file_name);
    fgets(line, LINE_LENGTH_MAX, fp);
    if (strstr(line, "general")) {
        isUnsy = 1;
    }
    do {
        fgets(line, LINE_LENGTH_MAX, fp);
    } while(line[0] == '%');
  
    /* Get size info */
    sscanf(line, "%d %d %d", M, N, nz);

    /* Store in COO format */
    num = 0;
    col_coo = (int *)malloc(sizeof(int) * (*nz));
    row_coo = (int *)malloc(sizeof(int) * (*nz));
    val_coo = (real *)malloc(sizeof(real) * (*nz));


    while (fgets(line, LINE_LENGTH_MAX, fp)) {
        ch = line;
        /* Read first word (row id)*/
        row_coo[num] = (int)(atoi(ch) - 1);
        ch = strchr(ch, ' ');
        ch++;
        /* Read second word (column id)*/
        col_coo[num] = (int)(atoi(ch) - 1);
        ch = strchr(ch, ' ');

        if (ch != NULL) {
            ch++;
            /* Read third word (value data)*/
            val_coo[num] = (real)atof(ch);
            ch = strchr(ch, ' ');
        }
        else {
            val_coo[num] = 1.0;
        }
        num++;
    }
    fclose(fp);

    /* Count the number of non-zero in each row */
    nnz_num = (int *)malloc(sizeof(int) * (*M));
    for (i = 0; i < (*M); i++) {
        nnz_num[i] = 0;
    }
    for (i = 0; i < num; i++) {
        nnz_num[row_coo[i]]++;
        if(col_coo[i] != row_coo[i] && isUnsy == 0) {
            nnz_num[col_coo[i]]++;
            (*nz)++;
        }
    }

    /* Allocation of rpt, col, val */
    rpt_ = (int *)malloc(sizeof(int) * ((*M) + 1));
    col_ = (int *)malloc(sizeof(int) * (*nz));
    val_ = (real *)malloc(sizeof(real) * (*nz));

    offset = 0;
    *nnz_max = 0;
    for (i = 0; i < (*M); i++) {
        rpt_[i] = offset;
        offset += nnz_num[i];
        if(*nnz_max < nnz_num[i]){
            *nnz_max = nnz_num[i];
        }
    }
    rpt_[(*M)] = offset;

    each_row_index = (int *)malloc(sizeof(int) * (*M));
    for (i = 0; i < (*M); i++) {
        each_row_index[i] = 0;
    }
  
    for (i = 0; i < num; i++) {
        col_[rpt_[row_coo[i]] + each_row_index[row_coo[i]]] = col_coo[i];
        val_[rpt_[row_coo[i]] + each_row_index[row_coo[i]]++] = val_coo[i];
    
        if (col_coo[i] != row_coo[i] && isUnsy==0) {
            col_[rpt_[col_coo[i]] + each_row_index[col_coo[i]]] = row_coo[i];
            val_[rpt_[col_coo[i]] + each_row_index[col_coo[i]]++] = val_coo[i];
        }
    }

    *rpt = rpt_;
    *col = col_;
    *val = val_;

    free(line);
    free(nnz_num);
    free(row_coo);
    free(col_coo);
    free(val_coo);
    free(each_row_index);

}

void init_csr_matrix_from_file(sfCSR *mat, char *file_name)
{
    convert_file_csr(file_name,
                     &(mat->rpt), &(mat->col), &(mat->val),
                     &(mat->M), &(mat->N), &(mat->nnz), &(mat->nnz_max));
    mat->matrix_name = file_name;
}

void csr_memcpy(sfCSR *mat)
{
    checkCudaErrors(cudaMalloc((void **)&(mat->d_rpt), sizeof(int) * (mat->M + 1)));
    checkCudaErrors(cudaMalloc((void **)&(mat->d_col), sizeof(int) * mat->nnz));
    checkCudaErrors(cudaMalloc((void **)&(mat->d_val), sizeof(real) * mat->nnz));
  
    checkCudaErrors(cudaMemcpy(mat->d_rpt, mat->rpt, sizeof(int) * (mat->M + 1), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(mat->d_col, mat->col, sizeof(int) * mat->nnz, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(mat->d_val, mat->val, sizeof(real) * mat->nnz, cudaMemcpyHostToDevice));

}

void csr_memcpyDtH(sfCSR *mat)
{
    mat->rpt = (int *)malloc(sizeof(int) * (mat->M + 1));
    mat->col = (int *)malloc(sizeof(int) * (mat->nnz));
    mat->val = (real *)malloc(sizeof(real) * (mat->nnz));
  
    checkCudaErrors(cudaMemcpy(mat->rpt, mat->d_rpt, sizeof(int) * (mat->M + 1), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(mat->col, mat->d_col, sizeof(int) * mat->nnz, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(mat->val, mat->d_val, sizeof(real) * mat->nnz, cudaMemcpyDeviceToHost));

}

/* Plan set */
void init_plan(sfPlan *plan)
{
    plan->isPlan = FALSE;
}

void set_plan(sfPlan *plan, size_t seg_size, int block_size)
{
    plan->isPlan = TRUE;
    plan->seg_size = seg_size;
    if (plan->seg_size > USHORT_MAX) {
        plan->seg_size = USHORT_MAX;
    }
    plan->block_size = block_size;
    if (plan->block_size < 1 || plan->block_size > MAX_BLOCK_SIZE) {
        plan->block_size = 1;
    }
}

/*Initializing vector*/
void init_vector(real *x, int row)
{
    int i;

    srand48((unsigned)time(NULL));

    for (i = 0; i < row; i++) {
        x[i] = drand48();
    }
}

/*Release memory object or free*/
void release_cpu_csr(sfCSR mat)
{
    free(mat.rpt);
    free(mat.col);
    free(mat.val);
}

void release_csr(sfCSR mat)
{
    cudaFree(mat.d_rpt);
    cudaFree(mat.d_col);
    cudaFree(mat.d_val);
}

void release_cpu_amb(sfAMB mat)
{
    free(mat.cs);
    free(mat.cl);
    free(mat.sellcs_val);
    free(mat.sellcs_col);
    free(mat.s_write_permutation);
    free(mat.s_write_permutation_offset);
}

void release_amb(sfAMB mat)
{
    cudaFree(mat.d_cs);
    cudaFree(mat.d_cl);
    cudaFree(mat.d_sellcs_val);
    cudaFree(mat.d_sellcs_col);
    cudaFree(mat.d_write_permutation);
    cudaFree(mat.d_s_write_permutation);
    cudaFree(mat.d_s_write_permutation_offset);
}

/*
 * Check answer of SpMV
 */
void csr_kernel(real *y, sfCSR *cpu_mat, real *x)
{
    int i, j;
    real ans;
    int M = cpu_mat->M;
    int *rpt, *col;
    real *val;

    rpt = cpu_mat->rpt;
    col = cpu_mat->col;
    val = cpu_mat->val;
  
    for (i = 0; i < M; i++) {
        ans = 0;
        for (j = 0; j < (rpt[i + 1] - rpt[i]); j++) {
            ans += val[rpt[i] + j] * x[col[rpt[i] + j]];
        }
        y[i] = ans;
    }
}

void ans_check(real *csr_ans, real *ans_vec, int N)
{
    int i;
    int total_fail = 10;
    real delta, base, scale;
  
    for (i = 0; i < N; i++) {
    
        delta = ans_vec[i] - csr_ans[i];
        base = ans_vec[i];

        if (delta < 0) {
            delta *= -1;
        }
        if (base < 0) {
            base *= -1;
        }
        
#ifdef FLOAT
        scale = 1000;
#else
        scale = 1000 * 1000;
#endif
        if (delta * 100 * scale > base) {
            printf("i=%d, ans=%e, csr=%e, delta=%e\n", i, ans_vec[i], csr_ans[i], delta);
            total_fail--;
            if(total_fail == 0)
                break;
        }
    }

    if (total_fail != 10){
        printf("Calculation Result is Incorrect\n");
    }
    else {
        printf("Calculation Result is Correct\n");
    }
}

void check_spgemm_answer(sfCSR c, sfCSR ans)
{
    int i;
    int M, nz;

    M = c.M;
    if (c.nnz != ans.nnz) {
        printf("nnz is not correct: %d (correct), %d (incorrect)\n", ans.nnz, c.nnz);
        return;
    }
    nz = c.nnz;

    /* check rpt */
    for (i = 0; i < M + 1; i++) {
        if (c.rpt[i] != ans.rpt[i]) {
            printf("rpt[%d] is not correct: %d (correct),%d (incorrect)\n", i, ans.rpt[i], c.rpt[i]);
            return;
        }
    }
    /* check col */
    for (i = 0; i < nz; i++) {
        if (c.col[i] != ans.col[i]) {
            printf("col[%d] is not correct: %d (correct), %d (incorrect)\n", i, ans.col[i], c.col[i]);
            return;
        }
    }
    /* check val */
    real delta, base, scale;
    int total_fail = 10;
#ifdef FLOAT
    scale = 1000;
#else
    scale = 1000 * 1000;
#endif
    for (i = 0; i < nz; i++) {
        delta = ans.val[i] - c.val[i];
        base = ans.val[i];
        if (delta < 0) delta *= -1;
        if (base < 0) base *= -1;

        if (delta * 1000 * scale > base) {
            printf("val[%d]: ans=%e, c=%e, delta=%e\n", i, ans.val[i], c.val[i], delta);
            total_fail--;
            if(total_fail == 0)
                break;
        }
    }
    if (total_fail != 10){
        printf("Calculation Result is Incorrect\n");
    }
    else {
        printf("Calculation Result is Correct\n");
    }
}


