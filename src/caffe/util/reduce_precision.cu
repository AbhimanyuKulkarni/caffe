#include <algorithm>

#include "caffe/common.hpp"
#include "caffe/util/im2col.hpp"

namespace caffe {

template <typename Dtype>
__global__ void reduce_precision_gpu_kernel(Dtype* data, size_t size, const unsigned int prec, const float scale) {
  CUDA_KERNEL_LOOP(index, size) {
    Dtype d = data[index];
    d = d * scale;
    Dtype dmax = (1 << (prec-1)) - 1; // e.g. 127 =  01111111
    Dtype dmin = - (1 << (prec-1));   //     -128 = -10000000
    if (d > dmax) d = dmax;
    if (d < dmin) d = dmin;
    d = Dtype(int(d));
    d = d / scale;
    data[index] = d;
  }
}

template <typename Dtype>
void reduce_precision_gpu(Dtype* data, size_t size, const unsigned int prec, const float scale) {
  size_t num_kernels = size;
  reduce_precision_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(data, size, prec, scale);
}

template void reduce_precision_gpu<float>(float* data, size_t size, const unsigned int prec, const float scale);
template void reduce_precision_gpu<double>(double* data, size_t size, const unsigned int prec, const float scale);


}  // namespace caffe
