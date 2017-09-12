#include <algorithm>

#include "caffe/common.hpp"
#include "caffe/util/im2col.hpp"

namespace caffe {

template <typename Dtype>
__global__ void reduce_precision_gpu_kernel(Dtype* data, size_t size, const unsigned int prec, const float scale) {
  CUDA_KERNEL_LOOP(index, size) {
    Dtype d = data[index];
    Dtype shift = 0.0;
    Dtype dmax =   (1 << (prec-1)) - 1 + shift; // e.g. prec = 2, dmin,dmax = -1,1
    Dtype dmin = - (1 << (prec-1)) + 1 + shift; 

    d *= scale;

    if (d > dmax) d = dmax;
    if (d < dmin) d = dmin;

    d += shift;
    d = Dtype(round(d));
    d -= shift;

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
