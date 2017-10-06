#include <vector>

#include "boost/shared_ptr.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/util/reduce_precision.hpp"

#ifndef MYDEBUG 
#define MYDEBUG 0
#endif

namespace caffe {

// patrickjudd: helper function for setting precision of blobs 
template <typename Dtype>
void reduce_precision_blob_gpu(Blob<Dtype> & blob, 
      const PrecisionParameter & param, bool diff, const char * name){

    if (param.precision() == 0)
      return;

#if MYDEBUG

    int start=0;
    int end=blob.count();
     std::cout << name << " " << param.precision() 
        << " " << param.scale() << std::endl;

    for (int i=start; i<end ; i++){
      std::cout << name << "\tIN\t" << i << "\t" << blob.cpu_data()[i] << std::endl;
    }

#endif

    // if precision is the default value (0), dont touch the blob
    // 0 is not a valid precision
    if (param.precision() != 0){
      /*std::cout << name << "\t reducing precision to " << param.precision() << " with scaling " << param.scale() << "\n";*/
      reduce_precision_gpu(
          (diff)? blob.mutable_gpu_diff() : blob.mutable_gpu_data(), 
          blob.count(), 
          param.precision(), 
          param.scale(),
          param.quantizer()
          );
    }
    /*std::cout << "CAFFE CPU DATA\t" << blob.cpu_data()[0] << std::endl;*/

#if MYDEBUG

    for (int i=start; i<end ; i++){
      std::cout << name << "\tOUT\t" << i << "\t" << blob.cpu_data()[i] << std::endl;
    }

#endif

}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();

  // patrickjudd: reduce precision of weights 
  reduce_precision_blob_gpu<Dtype>( *(this->blobs_[0]),  
      this->layer_param_.fwd_wgt_precision_param(), 0/*diff*/, "fwd_wgt");

  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();

    // patrickjudd: reduce precision of activations 
    reduce_precision_blob_gpu<Dtype>( *bottom[i],
        this->layer_param_.fwd_act_precision_param(), 0/*diff*/, "fwd_act");

    Dtype* top_data = top[i]->mutable_gpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->gpu_data();
        this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();

  // patrickjudd: reduce precision of weights
  reduce_precision_blob_gpu<Dtype>( *(this->blobs_[0]),
      this->layer_param_.bwd_wgt_precision_param(), 0/*diff*/, "bwd_wgt");

  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();

    // patrickjudd: reduce precision of activations and gradients
    reduce_precision_blob_gpu<Dtype>( *bottom[i],
        this->layer_param_.bwd_act_precision_param(), 0/*diff*/, "bwd_act");
    reduce_precision_blob_gpu<Dtype>( *top[i],
        this->layer_param_.bwd_grd_precision_param(), 1/*diff*/, "bwd_grd");

    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_gpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionLayer);

}  // namespace caffe
