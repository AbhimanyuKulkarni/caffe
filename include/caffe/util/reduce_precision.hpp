#ifndef _CAFFE_UTIL_REDUCE_PRECISION_HPP_
#define _CAFFE_UTIL_REDUCE_PRECISION_HPP_

namespace caffe {

template <typename Dtype>
void reduce_precision_gpu(Dtype* data, size_t size, const unsigned int prec, const float scale, const int quantizer);

}  // namespace caffe

#endif //_CAFFE_UTIL_REDUCE_PRECISION_HPP_
