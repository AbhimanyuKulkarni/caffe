#include <algorithm>

#include "caffe/common.hpp"
#include "caffe/util/im2col.hpp"

namespace caffe {

template <typename Dtype>
__global__ void reduce_precision_gpu_kernel(Dtype* data, size_t size, const unsigned int prec, const float scale, const int quantizer) {
  // quantizers 
  // 0: mid rise using all signed integers
  // 1: mid tread using all signed integers
  // 2: mid tread with a symmetric range (odd number of bins)
  // 3: 2 with a squished zero bin for only 0 values
  int midrise   = quantizer < 1;
  int odd       = quantizer > 1;
  int zero_bin  = quantizer > 2;
  int intmax;
  int intmin;

  intmax = (1 << (prec-1)) - 1;
  if (odd){
    intmin = -1 * intmax; // -1, 0, 1
  } else {
    intmin = - (1 << (prec-1)); // -2, -1, 0, 1
  }

  CUDA_KERNEL_LOOP(index, size) {
    Dtype d = data[index];
    int q;

    // move near zero values to first non zero bins
    if (zero_bin){
      if (d > 0 && d < scale) // is this right?
        d = scale;
      else if (d < 0 && d > -scale)
        d = -scale;
    }

    // quantize
    if (midrise) {
      q = floor(d * scale);
    } else {
      q = floor(d * scale + 0.5);
    }
    // clamp
    if (q > intmax) q = intmax;
    if (q < intmin) q = intmin;

    // reconstruct
    if (d == 0 && zero_bin) {
      d = 0;
    } else if (midrise) {
      d = ( q + 0.5 ) / scale;
    } else {
      d = ( q ) / scale;
    }

    data[index] = d;
  }
}

template <typename Dtype>
void reduce_precision_gpu(Dtype* data, size_t size, const unsigned int prec, const float scale, const int quantizer) {
  size_t num_kernels = size;
  reduce_precision_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(data, size, prec, scale, quantizer);
}

template void reduce_precision_gpu<float>(float* data, size_t size, const unsigned int prec, const float scale, const int quantizer);
template void reduce_precision_gpu<double>(double* data, size_t size, const unsigned int prec, const float scale, const int quantizer);


}  // namespace caffe
