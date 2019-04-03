CXX = nvcc
NVCC = nvcc

CFLAGS = -O3 -g
CFLAGS += -L. ${REAL} -lm
LDFLAGS = ${CFLAGS}

# for Device Code
CUDA_PATH = /usr/local/cuda
LDFLAGS += -L${CUDA_PATH}/lib64
LDFLAGS += -arch=sm_70 -lcudart -lcusparse
INCLUDE = -I./inc
INCLUDE += -I${CUDA_PATH}/include
INCLUDE += -I${CUDA_PATH}/samples/common/inc

BIN = ./bin
SRC = ./sample
OBJ = ./obj
INC = ./inc

OBJ_SUF = .o
OS_SUF = .s.o
OD_SUF = .d.o
TS_SUF = _s
TD_SUF = _d

SRC_SPMV = $(SRC)/spmv
SAMPLE_SPMV = $(wildcard $(SRC_SPMV)/*.cu)
SAMPLE_SPMV_TARGET = $(SAMPLE_SPMV:$(SRC)%=$(BIN)%)

SRC_SPGEMM = $(SRC)/spgemm
SAMPLE_SPGEMM = $(wildcard $(SRC_SPGEMM)/*.cu)
SAMPLE_SPGEMM_TARGET = $(SAMPLE_SPGEMM:$(SRC)%=$(BIN)%)

all :
	make spmv
	make spgemm

spmv: $(SAMPLE_SPMV_TARGET:.cu=$(TS_SUF)) $(SAMPLE_SPMV_TARGET:.cu=$(TD_SUF))
spgemm: $(SAMPLE_SPGEMM_TARGET:.cu=$(TS_SUF)) $(SAMPLE_SPGEMM_TARGET:.cu=$(TD_SUF))

$(BIN)/%$(TS_SUF): $(OBJ)/%$(OS_SUF)
	mkdir -p $(dir $@)
	$(NVCC) -o $@ $^ $(LDFLAGS) $(INCLUDE)

$(BIN)/%$(TD_SUF): $(OBJ)/%$(OD_SUF)
	mkdir -p $(dir $@)
	$(NVCC) -o $@ $^ $(LDFLAGS) $(INCLUDE)

$(OBJ)/%$(OS_SUF) : $(SRC)/%.cu
	mkdir -p $(dir $@)
	$(NVCC) -c -DFLOAT $(LDFLAGS) $(INCLUDE) -o $@ $<

$(OBJ)/%$(OD_SUF) : $(SRC)/%.cu
	mkdir -p $(dir $@)
	$(NVCC) -c -DDOUBLE $(LDFLAGS) $(INCLUDE) -o $@ $<

$(OBJ)/%$(OS_SUF) : $(SRC)/%.cpp
	mkdir -p $(dir $@)
	$(NVCC) -c -DFLOAT $(LDFLAGS) $(INCLUDE) -o $@ $<

$(OBJ)/%$(OD_SUF) : $(SRC)/%.cpp
	mkdir -p $(dir $@)
	$(NVCC) -c -DDOUBLE $(LDFLAGS) $(INCLUDE) -o $@ $<

clean :
	rm -rf $(BIN)/*
	rm -rf $(OBJ)/*
