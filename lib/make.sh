#!/usr/bin/env bash

#CUDA_PATH=/usr/local/cuda/
export CPATH=/usr/local/cuda/include/
export CUDA_PATH=/usr/local/cuda/
export PATH=/usr/local/cuda/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH
export CXXFLAGS="-std=c++11"
export CFLAGS="-std=c99"




#python setup.py build_ext --inplace
#rm -rf build

CUDA_ARCH="-gencode arch=compute_60,code=sm_60 "

# compile NMS
cd model/nms/src
echo "Compiling nms kernels by nvcc..."
nvcc -c -o nms_cuda_kernel.cu.o nms_cuda_kernel.cu \ -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPI $CUDA_ARCH

cd ../
python build.py

# compile roi_temporal_pooling
cd ../../
cd model/roi_temporal_pooling/src
echo "Compiling roi temporal pooling kernels by nvcc..."
nvcc -c -o roi_temporal_pooling_kernel.cu.o roi_temporal_pooling_kernel.cu \
	 -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC $CUDA_ARCH
cd ../
python build.py
