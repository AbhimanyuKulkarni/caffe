#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/reduce_precision.hpp"

#ifndef MYDEBUG 
#define MYDEBUG 0
#endif

namespace caffe {

// patrickjudd: helper function for setting precision of blobs 
template <typename Dtype>
void reduce_precision_blob_gpu(Blob<Dtype> & blob, 
      const PrecisionParameter & param, bool diff, const char * name, const bool quantize){

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
      reduce_precision_gpu(
          (diff)? blob.mutable_gpu_diff() : blob.mutable_gpu_data(), 
          blob.count(), 
          param.precision(), 
          param.scale(),
          param.quantizer(),
          quantize,
          1 //round
          );
    }

#if MYDEBUG

    for (int i=start; i<end ; i++){
      std::cout << name << "\tOUT\t" << i << "\t" << blob.cpu_data()[i] << std::endl;
    }

#endif

}
template <typename Dtype>
void InnerProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  //const Dtype* weight = this->blobs_[0]->gpu_data();
  Blob<Dtype> * weight = this->blobs_[0].get();

  bool reduce_storage = this->layer_param_.fwd_wgt_precision_param().store_reduced();
  //printf("reduce_storage = %d prec = %d\n",reduce_storage, this->layer_param_.fwd_wgt_precision_param().precision());
  if (!reduce_storage){ 
    // clip weights 
    reduce_precision_blob_gpu<Dtype>( *(this->blobs_[0]),  
        this->layer_param_.fwd_wgt_precision_param(), 0/*diff*/, "fwd_wgt_clip", 0/*quantize*/);

    weight = new Blob<Dtype>(this->blobs_[0]->shape());
    weight->CopyFrom(*(this->blobs_[0]));
  }

  // patrickjudd: reduce precision of weights 
  reduce_precision_blob_gpu<Dtype>( *weight,  
      this->layer_param_.fwd_wgt_precision_param(), 0/*diff*/, "fwd_wgt", 1);
  // patrickjudd: reduce precision of activations 
  reduce_precision_blob_gpu<Dtype>( *bottom[0],
      this->layer_param_.fwd_act_precision_param(), 0/*diff*/, "fwd_act", 1);

  if (M_ == 1) {
    caffe_gpu_gemv<Dtype>(CblasNoTrans, N_, K_, (Dtype)1.,
                         weight->gpu_data(), bottom_data, (Dtype)0., top_data);
    if (bias_term_)
      caffe_gpu_axpy<Dtype>(N_, bias_multiplier_.cpu_data()[0],
                            this->blobs_[1]->gpu_data(), top_data);
  } else {
    caffe_gpu_gemm<Dtype>(CblasNoTrans,
                          transpose_ ? CblasNoTrans : CblasTrans,
                          M_, N_, K_, (Dtype)1.,
                          bottom_data, weight->gpu_data(), (Dtype)0., top_data);
    if (bias_term_)
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
                            bias_multiplier_.gpu_data(),
                            this->blobs_[1]->gpu_data(), (Dtype)1., top_data);
  }
  if (!reduce_storage){
    delete weight;
  }
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  Blob<Dtype> * weight = this->blobs_[0].get();

  bool reduce_storage = this->layer_param_.fwd_wgt_precision_param().store_reduced();
  if (!reduce_storage){ // TODO make parameter to reduce storage  
    weight = new Blob<Dtype>(this->blobs_[0]->shape());
    weight->CopyFrom(*(this->blobs_[0]));
  }

  // patrickjudd: reduce precision of weights
  reduce_precision_blob_gpu<Dtype>( *weight,
      this->layer_param_.bwd_wgt_precision_param(), 0/*diff*/, "bwd_wgt", 1);
  // patrickjudd: reduce precision of activations and gradients
  reduce_precision_blob_gpu<Dtype>( *bottom[0],
      this->layer_param_.bwd_act_precision_param(), 0/*diff*/, "bwd_act", 1);
  reduce_precision_blob_gpu<Dtype>( *top[0],
      this->layer_param_.bwd_grd_precision_param(), 1/*diff*/, "bwd_grd", 1);

  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    // Gradient with respect to weight
    if (transpose_) {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          K_, N_, M_,
          (Dtype)1., bottom_data, top_diff,
          (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
    } else {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          N_, K_, M_,
          (Dtype)1., top_diff, bottom_data,
          (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bias
    caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.gpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_gpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bottom data
    if (transpose_) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, weight->gpu_data(),
          (Dtype)0., bottom[0]->mutable_gpu_diff());
    } else {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          M_, K_, N_,
         (Dtype)1., top_diff, weight->gpu_data(),
         (Dtype)0., bottom[0]->mutable_gpu_diff());
    }
  }
  if (!reduce_storage){
    delete weight;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(InnerProductLayer);

}  // namespace caffe
