#######################################################################################################
# Setup clBLAS for LINUX
# wget https://github.com/clMathLibraries/clBLAS/releases/download/v2.12/clBLAS-2.12.0-Linux-x64.tar.gz
# tar zxf clBLAS-2.12.0-Linux-x64.tar.gz
# cd clBLAS-2.12.0-Linux-x64
# cp bin/* /usr/bin/
# cp lib64/* /usr/lib64/
# cp -r lib64/cmake/* /usr/lib64/cmake/
# cp -r lib64/pkgconfig/* /usr/lib64/pkgconfig/
# cp -r include/* /usr/include/
#

GPU=1
OPENCV=1
DEBUG=0

#ARCH= -gencode arch=compute_30,code=sm_30 \
#      -gencode arch=compute_35,code=sm_35 \
#      -gencode arch=compute_50,code=[sm_50,compute_50] \
#      -gencode arch=compute_52,code=[sm_52,compute_52]
#     -gencode arch=compute_20,code=[sm_20,sm_21] \ This one is deprecated?

# This is what I use, uncomment if you know your arch and want to specify
# ARCH= -gencode arch=compute_52,code=compute_52

VPATH=./src/:./examples
SLIB=libdarknet.so
ALIB=libdarknet.a
EXEC=darknet
OBJDIR=./obj/

CC=gcc
AR=ar
ARFLAGS=rcs
OPTS=
LDFLAGS= -lm -lz -lpthread
COMMON= -Iinclude/ -Isrc/
CFLAGS=-Wall -Wno-unknown-pragmas -Wno-unused-variable -Wfatal-errors -fPIC

ifeq ($(DEBUG), 1)
OPTS=-O0 -g
else
OPTS=-O2 -mfpmath=sse
endif

CFLAGS+=$(OPTS)

ifeq ($(OPENCV), 1) 
COMMON+= -DOPENCV
CFLAGS+= -DOPENCV
LDFLAGS+= `pkg-config --libs opencv` 
COMMON+= `pkg-config --cflags opencv` 
endif

ifeq ($(GPU), 1) 
COMMON+= -DGPU -DOPENCL -DCONFIG_USE_DOUBLE
CFLAGS+= -DGPU -DOPENCL -DCONFIG_USE_DOUBLE -I/usr/include/ -I/usr/local/cuda/include/
LDFLAGS+= -L/usr/local/cuda/lib64 -lOpenCL -L/usr/lib64 -lclBLAS
LDFLAGS+= -L/usr/lib64
endif

OBJ=gemm.o utils.o opencl.o deconvolutional_layer.o convolutional_layer.o list.o image.o activations.o im2col.o col2im.o blas.o crop_layer.o dropout_layer.o maxpool_layer.o softmax_layer.o data.o matrix.o network.o connected_layer.o cost_layer.o parser.o option_list.o detection_layer.o route_layer.o box.o normalization_layer.o avgpool_layer.o layer.o local_layer.o shortcut_layer.o activation_layer.o rnn_layer.o gru_layer.o crnn_layer.o demo.o batchnorm_layer.o region_layer.o reorg_layer.o tree.o  lstm_layer.o yolo_layer.o upsample_layer.o logistic_layer.o l2norm_layer.o
EXECOBJA=captcha.o lsd.o super.o art.o tag.o cifar.o go.o rnn.o segmenter.o regressor.o classifier.o coco.o yolo.o detector.o nightmare.o attention.o darknet.o
ifeq ($(GPU), 1) 
#LDFLAGS+= -lstdc++
OBJ+=convolutional_kernels.o deconvolutional_kernels.o activation_kernels.o im2col_kernels.o col2im_kernels.o blas_kernels.o crop_layer_kernels.o dropout_layer_kernels.o maxpool_layer_kernels.o avgpool_layer_kernels.o
else
#LDFLAGS+= -lstdc++
OBJ+=cpu.o
endif

EXECOBJ = $(addprefix $(OBJDIR), $(EXECOBJA))
OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile include/darknet.h

#all: obj backup results $(SLIB) $(ALIB) $(EXEC)
all: obj  results $(SLIB) $(ALIB) $(EXEC)


$(EXEC): $(EXECOBJ) $(ALIB)
	$(CC) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(ALIB)

$(ALIB): $(OBJS)
	$(AR) $(ARFLAGS) $@ $^

$(SLIB): $(OBJS)
	$(CC) $(CFLAGS) -shared $^ -o $@ $(LDFLAGS)

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

obj:
	mkdir -p obj
backup:
	mkdir -p backup
results:
	mkdir -p results

.PHONY: clean

clean:
	rm -rf $(OBJS) $(SLIB) $(ALIB) $(EXEC) $(EXECOBJ)

