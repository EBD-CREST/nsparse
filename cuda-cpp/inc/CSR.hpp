#include <iostream>
#include <string>
#include <cuda.h>

using namespace std;

#ifndef CSR_H
#define CSR_H
template <class idType, class valType>
class CSR
{
public:
    CSR():nrow(0), ncolumn(0), nnz(0), devise_malloc(false)
    {
    }
    ~CSR()
    {
    }
    void release_cpu_csr()
    {
        delete[] rpt;
        delete[] colids;
        delete[] values;
    }
    void release_csr()
    {
        if (devise_malloc) {
            cudaFree(d_rpt);
            cudaFree(d_colids);
            cudaFree(d_values);
        }
        devise_malloc = false;
    }
    void init_data_from_mtx(string file_path);
    void memcpyHtD()
    {
        if (!devise_malloc) {
            cout << "Allocating memory space for matrix data on devise memory" << endl;
            cudaMalloc((void **)&d_rpt, sizeof(idType) * (nrow + 1));
            cudaMalloc((void **)&d_colids, sizeof(idType) * nnz);
            cudaMalloc((void **)&d_values, sizeof(valType) * nnz);
        }
        cout << "Copying matrix data to GPU devise" << endl;
        cudaMemcpy(d_rpt, rpt, sizeof(idType) * (nrow + 1), cudaMemcpyHostToDevice);
        cudaMemcpy(d_colids, colids, sizeof(idType) * nnz, cudaMemcpyHostToDevice);
        cudaMemcpy(d_values, values, sizeof(valType) * nnz, cudaMemcpyHostToDevice);
        devise_malloc = true;
    }
    void memcpyDtH()
    {
        rpt = new idType[nrow + 1];
        colids = new idType[nnz];
        values = new valType[nnz];
        cout << "Matrix data is copied to Host" << endl;
        cudaMemcpy(d_rpt, rpt, sizeof(idType) * (nrow + 1), cudaMemcpyDeviceToHost);
        cudaMemcpy(d_colids, colids, sizeof(idType) * nnz, cudaMemcpyDeviceToHost);
        cudaMemcpy(d_values, values, sizeof(valType) * nnz, cudaMemcpyDeviceToHost);
    }

    void spmv_cpu(valType *x, valType *y);
    
    idType *rpt;
    idType *colids;
    valType *values;
    idType *d_rpt;
    idType *d_colids;
    valType *d_values;
    idType nrow;
    idType ncolumn;
    idType nnz;
    bool host_malloc;
    bool devise_malloc;
};

template <class idType, class valType>
void CSR<idType, valType>::init_data_from_mtx(string file_path)
{
    int i, num;
    int isUnsy;
    char *line, *ch;
    FILE *fp;
    idType *col_coo, *row_coo, *nnz_num, *each_row_index;
    valType *val_coo;
    int LINE_LENGTH_MAX = 256;

    devise_malloc = false;
    
    isUnsy = 0;
    line = new char[LINE_LENGTH_MAX];
  
    /* Open File */
    fp = fopen(file_path.c_str(), "r");
    if (fp == NULL) {
        cout << "Cannot find file" << endl;
        exit(1);
    }

    fgets(line, LINE_LENGTH_MAX, fp);
    if (strstr(line, "general")) {
        isUnsy = 1;
    }
    do {
        fgets(line, LINE_LENGTH_MAX, fp);
    } while(line[0] == '%');
  
    /* Get size info */
    sscanf(line, "%d %d %d", &nrow, &ncolumn, &nnz);
    
    /* Store in COO format */
    num = 0;
    col_coo = new idType[nnz];
    row_coo = new idType[nnz];
    val_coo = new valType[nnz];

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
            val_coo[num] = (valType)atof(ch);
            ch = strchr(ch, ' ');
        }
        else {
            val_coo[num] = 1.0;
        }
        num++;
    }
    fclose(fp);
    delete[] line;

    /* Count the number of non-zero in each row */
    nnz_num = new idType[nrow];
    for (i = 0; i < nrow; i++) {
        nnz_num[i] = 0;
    }
    for (i = 0; i < num; i++) {
        nnz_num[row_coo[i]]++;
        if(col_coo[i] != row_coo[i] && isUnsy == 0) {
            nnz_num[col_coo[i]]++;
            nnz++;
        }
    }

    /* Allocation of rpt, col, val */
    rpt = new idType[nrow + 1];
    colids = new idType[nnz];
    values = new valType[nnz];

    rpt[0] = 0;
    for (i = 0; i < nrow; i++) {
        rpt[i + 1] = rpt[i] + nnz_num[i];
    }

    each_row_index = new idType[nrow];
    for (i = 0; i < nrow; i++) {
        each_row_index[i] = 0;
    }
  
    for (i = 0; i < num; i++) {
        colids[rpt[row_coo[i]] + each_row_index[row_coo[i]]] = col_coo[i];
        values[rpt[row_coo[i]] + each_row_index[row_coo[i]]++] = val_coo[i];
    
        if (col_coo[i] != row_coo[i] && isUnsy==0) {
            colids[rpt[col_coo[i]] + each_row_index[col_coo[i]]] = row_coo[i];
            values[rpt[col_coo[i]] + each_row_index[col_coo[i]]++] = val_coo[i];
        }
    }

    cout << "Row: " << nrow << ", Column: " << ncolumn << ", Nnz: " << nnz << endl;

    delete[] nnz_num;
    delete[] row_coo;
    delete[] col_coo;
    delete[] val_coo;
    delete[] each_row_index;

}

template <class idType, class valType>
void CSR<idType, valType>::spmv_cpu(valType *x, valType *y)
{
    idType i, j;
    valType ans;
  
    for (i = 0; i < nrow; ++i) {
        ans = 0;
        for (j = 0; j < (rpt[i + 1] - rpt[i]); j++) {
            ans += values[rpt[i] + j] * x[colids[rpt[i] + j]];
        }
        y[i] = ans;
    }
}

#endif
