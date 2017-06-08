CXX = g++
NVCC = nvcc

CFLAGS = -O3 -g
CFLAGS += -L. ${REAL} -lm
LDFLAGS = ${CFLAGS}

# for Device Code
CUDA_PATH = /usr/local/cuda
LDFLAGS += -L${CUDA_PATH}/lib64
LDFLAGS += -lm -arch=sm_52 -lcudart -lcusparse
INCLUDE = -I./inc
INCLUDE += -I${CUDA_PATH}/include
INCLUDE += -I${CUDA_PATH}/samples/common/inc

BIN = ./bin
SRC = ./src
OBJ = ./obj

OBJ_SUF = .o
OS_SUF = .s.o
OD_SUF = .d.o

OBJS_LIB = $(OBJ)/nsparse.o
OBJS_CU_CSR = $(OBJS_LIB) $(OBJ)/sample/spmv/spmv_cu_csr.o 
OBJS_AMB = $(OBJS_LIB) $(OBJ)/conversion/convert_amb.o $(OBJ)/kernel/kernel_spmv_amb.o $(OBJ)/sample/spmv/spmv_amb.o 

all :
	make amb

spmv : 
	make cu_csr
	make amb

cu_csr :
	make cu_csr_s
	make cu_csr_d

cu_csr_s : $(OBJS_CU_CSR:$(OBJ_SUF)=$(OS_SUF))
	mkdir -p $(BIN)
	$(NVCC) -o $(BIN)/$@ $(OBJS_CU_CSR:$(OBJ_SUF)=$(OS_SUF)) $(LDFLAGS) $(INCLUDE)

cu_csr_d : $(OBJS_CU_CSR:$(OBJ_SUF)=$(OD_SUF))
	mkdir -p $(BIN)
	$(NVCC) -o $(BIN)/$@ $(OBJS_CU_CSR:$(OBJ_SUF)=$(OD_SUF)) $(LDFLAGS) $(INCLUDE)

amb :
	make amb_s
	make amb_d

amb_s : $(OBJS_AMB:$(OBJ_SUF)=$(OS_SUF))
	mkdir -p $(BIN)
	$(NVCC) -o $(BIN)/$@ $(OBJS_AMB:$(OBJ_SUF)=$(OS_SUF)) $(LDFLAGS) $(INCLUDE)

amb_d : $(OBJS_AMB:$(OBJ_SUF)=$(OD_SUF))
	mkdir -p $(BIN)
	$(NVCC) -o $(BIN)/$@ $(OBJS_AMB:$(OBJ_SUF)=$(OD_SUF)) $(LDFLAGS) $(INCLUDE)

$(OBJ)/%$(OS_SUF) : $(SRC)/%.c
	mkdir -p $(dir $@)
	$(CXX) -c -DFLOAT $(CFLAGS) $(INCLUDE) -o $@ $<

$(OBJ)/%$(OD_SUF) : $(SRC)/%.c
	mkdir -p $(dir $@)
	$(CXX) -c -DDOUBLE $(CFLAGS) $(INCLUDE) -o $@ $<

$(OBJ)/%$(OS_SUF) : $(SRC)/%.cu
	mkdir -p $(dir $@)
	$(NVCC) -c -DFLOAT $(LDFLAGS) $(INCLUDE) -o $@ $<

$(OBJ)/%$(OD_SUF) : $(SRC)/%.cu
	mkdir -p $(dir $@)
	$(NVCC) -c -DDOUBLE $(LDFLAGS) $(INCLUDE) -o $@ $<

clean :
	rm -rf $(BIN)/*
	rm -rf $(OBJ)/*
