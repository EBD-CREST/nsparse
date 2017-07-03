#include <cusparse_v2.h>

#ifdef FLOAT
typedef float real;

#elif defined DOUBLE
typedef double real;

#else
typedef double real;
#endif

#define div_round_up(a, b) ((a % b == 0)? a / b : a / b + 1)

/* Hardware Specific Parameters */
#define WARP_BIT 5
#define WARP 32
#define MAX_LOCAL_THREAD_NUM 1024
#define MAX_THREAD_BLOCK (MAX_LOCAL_THREAD_NUM / WARP)

/* Number of SpMV Execution for Evaluation or Test */
#define TRI_NUM 101
#define TEST_NUM 2

/* Number of SpGEMM Execution for Evaluation or Test */
#define SPGEMM_TRI_NUM 11

/* Define 2 related */
#define sfFLT_MAX 1000000000
#define SHORT_MAX 32768
#define SHORT_MAX_BIT 15
#define USHORT_MAX 65536
#define USHORT_MAX_BIT 16

#define SCL_BORDER 16
#define SCL_BIT ((1 << SCL_BORDER) - 1)

#define MAX_BLOCK_SIZE 20

/* Check the answer */
#define sfDEBUG


typedef enum
{
    FALSE,
    TRUE
} BOOL;

typedef struct
{
    size_t thread_grid;
    size_t thread_block;
    BOOL isPlan;
    int SIGMA;
    size_t seg_size;
    size_t seg_num;
    int block_size;
} sfPlan;

/* Structure of Formats*/
typedef struct
{
    int *rpt;
    int *col;
    real *val;
    int *d_rpt;
    int *d_col;
    real *d_val;
    int M;
    int N;
    int nnz;
    int nnz_max;
    char *matrix_name;
} sfCSR;


typedef struct
{
    int *cs;
    unsigned int *cl;
    unsigned short *sellcs_col;
    real *sellcs_val;
    unsigned short *s_write_permutation;
    unsigned short *s_write_permutation_offset;
    int *write_permutation;
    int *d_cs;
    unsigned int *d_cl;
    unsigned short *d_sellcs_col;
    real *d_sellcs_val;
    unsigned short *d_s_write_permutation;
    unsigned short *d_s_write_permutation_offset;
    int *d_write_permutation;
    int block_size;
    int nnz;
    int M;
    int N;
    int pad_M;
    int chunk;
    int SIGMA;
    int group_num_col;
    int nnz_max;
    int c_size;
    size_t seg_size;
    size_t seg_num;
    char *matrix_name;
} sfAMB;

/* Structure for SpGEMM */
typedef struct {
    cudaStream_t *stream;
    int *bin_size;
    int *bin_offset;
    int *d_bin_size;
    int *d_bin_offset;
    int *d_row_nz;
    int *d_row_perm;
    int max_intprod;
    int max_nz;
    int *d_max;
} sfBIN;

/*
 * Initialize
 */
void init_vector(real *x, int row);

void init_csr_matrix_from_file(sfCSR *mat, char *file_name);
void csr_memcpy(sfCSR *mat);
void csr_memcpyDtH(sfCSR *mat);

/*
 * Release MemObjects of Each Format structure
 */
void release_cpu_csr(sfCSR mat);
void release_cpu_amb(sfAMB mat);
void release_csr(sfCSR mat);
void release_amb(sfAMB mat);

/*
 * Converting matrix to AMB format
 */
void init_plan(sfPlan *plan);
void set_plan(sfPlan *plan, size_t seg_size, int block_size);
void sf_csr2amb(sfAMB *mat, sfCSR *csr_mat, real *d_x, sfPlan *plan);

/*
 * SpMV Kernel
 */
void csr_ans_check(real *val, int *col, int *rpt, real *rhs_vec, real *csr_ans, int N);
void ans_check(real *csr_ans, real *ans_vec, int N);
void csr_kernel(real *csr_ans, sfCSR *cpu_mat, real *rhs_vec);
void sf_spmv_amb(real *d_y, sfAMB *mat, real *d_x, sfPlan *plan);

/*
 * SpGEMM Kernel
 */
void get_spgemm_flop(sfCSR *a, sfCSR *b,
                     int M, long long int *flop);
void spgemm_kernel_cu_csr(sfCSR *a, sfCSR *b, sfCSR *c,
                          cusparseHandle_t *cusparseHandle,
                          cusparseOperation_t *trans_a,
                          cusparseOperation_t *trans_b,
                          cusparseMatDescr_t *descr_a,
                          cusparseMatDescr_t *descr_b);
void spgemm_cu_csr(sfCSR *a, sfCSR *b, sfCSR *c);
void check_spgemm_answer(sfCSR c, sfCSR ans);
void spgemm_kernel_hash(sfCSR *a, sfCSR *b, sfCSR *c);




