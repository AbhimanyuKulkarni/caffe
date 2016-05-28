#include <limits>
#include <cstdlib>
#include <stdio.h>
#include <assert.h>

#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>
#include <boost/random/uniform_int_distribution.hpp>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#include "caffe/util/error.hpp"
#include "caffe/util/math_functions.hpp"

<<<<<<< HEAD
#define GEN_HISTO_ 0
#define COUNT_DATA_ 0

=======
>>>>>>> 7c8c68eca96cc3284f8dd2e7920b6e254063b65e
namespace caffe {

enum eERROR_TYPE {
  NORMAL = 0,
  ADD_LOGNORMAL = 1,
  SUB_LOGNORMAL = 2,
  BITFLIP_SEF = 3,
  BITFLIP_EF = 4,
  BITFLIP_F = 5,
};

<<<<<<< HEAD
template <typename Dtype>
void Approximator<Dtype>::print_stats(string name) {
  stats_.print(name.c_str());
  if (GEN_HISTO) {
    stats_.print_histo(name.c_str());
  }
}

template <typename Dtype>
void Approximator<Dtype>::pre_layer (
      const vector<Blob<Dtype>*> & bottom, 
      const vector<Blob<Dtype>*> & top,
      vector<shared_ptr<Blob<Dtype> > > & weights,
      LayerParameter & param
      ) 
{
  
    if (GEN_HISTO_) {
      for (int bottom_id = 0; bottom_id < bottom.size(); ++bottom_id) {
        Dtype const * data = bottom[bottom_id]->cpu_data();
        const int count = bottom[bottom_id]->count();
        stats_.add(data, count);
      }
    }
#if 0 // do this in each particular layer?
    if (COUNT_DATA_) {
      const string name = param.name();
      const LayerParameter_LayerType& type_id = param.type();
      string type;
      stringstream layer_info, bottom_info, weight_info(""), param_info("");
      if ( bottom.size() > 0 ) {
        const Blob<Dtype>* b = bottom[0];
        bottom_info << " data(n c h w),"    <<  b->num() << "," <<  b->channels() << "," <<  b->height() << "," <<  b->width() << ",";
      }
      if ( weights.size() ) { 
        const boost::shared_ptr<Blob<Dtype> > w = weights[0];
        weight_info << " weight(n c h w),"  <<  w->num() << "," <<  w->channels() << "," <<  w->height() << "," <<  w->width() << ",";
      }
      
      switch (type_id) {
        case LayerParameter_LayerType_CONVOLUTION: {
          ConvolutionParameter param = param.convolution_param();
          type="conv";
          param_info << " param(kernel_size pad stride)," 
            << param.kernel_size() << ","
            << ( (param.has_pad())? param.pad() : 0 ) << ","
            << param.stride() << ",";
          break;
        } case LayerParameter_LayerType_INNER_PRODUCT: {
          type="fc";        
          break;
        } case LayerParameter_LayerType_LRN: {            
          param_info << " param(local_size),"
            << param.lrn_param().local_size() << ",";
          type="local";       
          break;
        } case LayerParameter_LayerType_POOLING: {       
          PoolingParameter param = param.pooling_param();
          param_info << " param(kernel_size pad stride pooling)," 
            << param.kernel_size() << ","
            << ( (param.has_pad())? param.pad() : 0 ) << ","
            << param.stride() << ","
            << ( (param.pool() == PoolingParameter_PoolMethod_MAX)? "max" : "avg") << ",";
          type="pool";      
          break;
        } case LayerParameter_LayerType_RELU: {      
          type="relu";      
          break;
        }
        //case LayerParameter_LayerType_DROPOUT:        type="drop";      break;
        default:                                      type="unknown";   break;                
      }
      layer_info << " layer(name type)," << name << "," << type << ",";

      LOG(INFO) << "ALL " << layer_info.str() << bottom_info.str() << weight_info.str() << param_info.str();
      int count_d = 0;
      for (int bottom_id = 0; bottom_id < bottom.size(); ++bottom_id) {
        const Blob<Dtype>* b = bottom[bottom_id];
        const int count = b->count();
        count_d += count;
        LOG(INFO) << "bottomshape (name,type,i,n,c,h,w):" << name << "," << type << "," << bottom_id << "," <<  b->num() << "," <<  b->channels() << "," <<  b->height() << "," <<  b->width();
      }
      int count_do = 0;
      for (int top_id = 0; top_id < top.size(); ++top_id) {
        const Blob<Dtype>* t = top[top_id];
        LOG(INFO) << "topshape (name,i,n,c,h,w):" << name << "," << type << "," << top_id << "," <<  t->num() << "," <<  t->channels() << "," <<  t->height() << "," <<  t->width();
        const int count = top[top_id]->count();
        count_do += count;
      }
      int count_w = 0;
      if ( weights.size() ) { 
        const boost::shared_ptr<Blob<Dtype> > w = weights[0];
        count_w = w->count();
        LOG(INFO) << "weightshape (name,n,c,h,w):" << name << "," << type << "," <<  w->num() << "," <<  w->channels() << "," <<  w->height() << "," <<  w->width();
      }

      LOG(INFO) << "datasize (name,datain,dataout,data,weight):" << name << "," << count_d << ", " << count_do << "," << count_d + count_do << "," << count_w;
      LOG(INFO) << "bottomsize (name,size):" << name << "," <<  bottom.size();
      LOG(INFO) << "topsize (name,size):" << name << "," <<  top.size();
    }
#endif

    // limit data magnitude
    if (    param.has_max_data_mag()
        &&  param.has_data_precision()
        ) {
      int mag = param.max_data_mag();
      int prec = param.data_precision();
      for (int bottom_id = 0; bottom_id < bottom.size(); ++bottom_id) {
        const int count = bottom[bottom_id]->count();
        Dtype* data = bottom[bottom_id]->mutable_cpu_data();
        limit_mag_prec (data, count, mag, prec); 
      }
    } else if (param.has_max_data_mag()) {
      int mag = param.max_data_mag();
      for (int bottom_id = 0; bottom_id < bottom.size(); ++bottom_id) {
        const int count = bottom[bottom_id]->count();
        Dtype* data = bottom[bottom_id]->mutable_cpu_data();
        limit_mag (data, count, mag);
      }
    } else if (param.has_data_precision()){
      int prec = param.data_precision();
      for (int bottom_id = 0; bottom_id < bottom.size(); ++bottom_id) {
        const int count = bottom[bottom_id]->count();
        Dtype* data = bottom[bottom_id]->mutable_cpu_data();
        limit_prec (data, count, prec);
      }
    }

    // limit weight magnitude
    if (param.has_max_weight_mag()){
      int mag = param.max_weight_mag();
      if ( weights.size() ) { 
        int count = weights[0]->count();
        Dtype* weight = weights[0]->mutable_cpu_data();
        limit_mag (weight, count, mag);
      }
    }

    // limit weight precision
    if (param.has_weight_precision()){
      int prec = param.weight_precision();
      if ( weights.size() ) { 
        int count = weights[0]->count();
        Dtype* weight = weights[0]->mutable_cpu_data();
        limit_prec (weight, count, prec);
      }
    }
    
    // end Patrick
}
template
void Approximator<float>::pre_layer ( const vector<Blob<float>*> & bottom, const vector<Blob<float>*> & top, vector<shared_ptr<Blob<float> > > & weights, LayerParameter & param);
template
void Approximator<double>::pre_layer ( const vector<Blob<double>*> & bottom, const vector<Blob<double>*> & top, vector<shared_ptr<Blob<double> > > & weights, LayerParameter & param);

// add_error
// adds error to a list of floating point values
// in: data
// size: number of elements in "in"
// type: type of error, see eERROR_TYPE
// error: error scaling parameter, typically standard deviation

template <typename Dtype>
void Approximator<Dtype>::add_error( Dtype * const in, int size, int type, Dtype error){
  //printf("add_error(float*, %d, %d, %f)\n", size, type, error);
  //histogram[0]++;  

  //for (int i=0; i<size; i++){
  //  printf ("%f\n",in[i]);
  //}

  if (error > 0.0) {
    CHECK_LT(error, 1);
=======
//extern long long int histogram[];

template <typename Dtype>
void add_error( Dtype * const in, int size) {

  char * error_str = getenv("CAFFE_ERROR");
  assert(error_str != 0);
  Dtype error = (Dtype) atof(error_str);
  //histogram[0]++;  

  for (int i=0; i<size; i++){
    printf ("%f\n",in[i]);
  }

  if (error > 0.0) {
    CHECK_LT(error, 1);
    char * error_type_str = getenv("CAFFE_ERROR_TYPE");
    assert(error_type_str != 0);
    eERROR_TYPE error_type = static_cast<eERROR_TYPE>(atoi(error_type_str));
>>>>>>> 7c8c68eca96cc3284f8dd2e7920b6e254063b65e

    int errorSize = size; // size of error matrix should match output of gemm
    

<<<<<<< HEAD
    switch (type) {
=======
    switch (error_type) {
>>>>>>> 7c8c68eca96cc3284f8dd2e7920b6e254063b65e
      case NORMAL:
      case ADD_LOGNORMAL:
      case SUB_LOGNORMAL:
        {
          Dtype * errorM = new Dtype[errorSize]; // matrix of error values
          Dtype * output_data = in;

          // get range of data values (min,max)
          Dtype max = std::numeric_limits<Dtype>::min();
          Dtype min = std::numeric_limits<Dtype>::max();
          int imax = -1;
          int imin = -1;
          for (int im=0; im < errorSize; im++){
            if (output_data[im] > max){
              max = output_data[im];
              imax = im;
            }
            if (output_data[im] < min){
              min = output_data[im];
              imin = im;
            }
          }
          Dtype diff = max - min;

          // generate error matrix with standard deviation = range * error
          Dtype std_dev = diff * error;
          Dtype lambda = 1.0 / std_dev;
<<<<<<< HEAD
          switch (type) {
=======
          switch (error_type) {
>>>>>>> 7c8c68eca96cc3284f8dd2e7920b6e254063b65e
            case NORMAL:
              caffe_rng_gaussian<Dtype>(errorSize, 0, std_dev, errorM); 
              caffe_add(errorSize, output_data, errorM, output_data);
              break;
            case ADD_LOGNORMAL:
              caffe_rng_exponential<Dtype>(errorSize, lambda, errorM); 
              caffe_add(errorSize, output_data, errorM, output_data);
              break;
            case SUB_LOGNORMAL:
              caffe_rng_exponential<Dtype>(errorSize, lambda, errorM); 
              caffe_sub(errorSize, output_data, errorM, output_data);
              break;
            case BITFLIP_SEF:
            case BITFLIP_EF:
            case BITFLIP_F:
              // fallthrough
            default:
<<<<<<< HEAD
              printf("Unsupported error_type %d\n", type);
=======
              printf("Unsupported error_type %d\n", error_type);
>>>>>>> 7c8c68eca96cc3284f8dd2e7920b6e254063b65e
              assert(false);
              break;
          }


          // add error matrix to data
          // printf("errorSize = %d\n",errorSize);
          // printf("max = data[%d] = %f, min = data[%d] = %f\n", imax, max, imin, min);
          // printf("output_data_rng_max = %f \n", output_data[imax]);
          break;
        }
      case BITFLIP_SEF:
      case BITFLIP_EF:
      case BITFLIP_F:
        {
<<<<<<< HEAD
=======

>>>>>>> 7c8c68eca96cc3284f8dd2e7920b6e254063b65e
          // initialize poisson RNG
          Dtype lambda = 1/error - 1;
          CHECK_GT(lambda, 0);
          boost::poisson_distribution<int> poisson_dist(lambda);
          boost::variate_generator<caffe::rng_t*, boost::poisson_distribution<int> >
            poisson_vg(caffe_rng(), poisson_dist);

          // initialize uniform RNG
          int num_bits;
<<<<<<< HEAD
          switch (type) {
=======
          switch (error_type) {
>>>>>>> 7c8c68eca96cc3284f8dd2e7920b6e254063b65e
            case BITFLIP_SEF: num_bits=32; break; 
            case BITFLIP_EF:  num_bits=31; break;
            case BITFLIP_F:   num_bits=23; break;
          }
          boost::random::uniform_int_distribution<> uniform_dist(0, num_bits - 1);
          boost::variate_generator<caffe::rng_t*, boost::random::uniform_int_distribution<> >
            uniform_vg(caffe_rng(), uniform_dist);

          for (int i=0; i < size; i++) {
            i += poisson_vg(); // goto the next randomly selected element
            if (i >= size) continue;
            // pick a bit to flip
            int bit = uniform_vg();
            int mask = 1 << bit;
            // warning: this only works for floats
            // should make this a function template
            union {float f; int i;} val;
            val.f = (float)in[i];
            //printf("i=%d\t bit=%d\n", i, bit);
            //printf("val.f=%f\n", val.f);
            //printf("val.i=%x\n", val.i);
            //printf("mask=%x\n", mask);
            val.i ^= mask;
            //printf("val.i=%x\n", val.i);
            //printf("val.f=%f\n", val.f);
            in[i] = (Dtype)val.f;
          }


          break;
        }
      default:
<<<<<<< HEAD
        printf("Unknown error type: %d\n", type);
=======
        printf("Unknown error type: %d\n", error_type);
>>>>>>> 7c8c68eca96cc3284f8dd2e7920b6e254063b65e
        assert(false);
        break;
    } // switch error_type
  } //if error > 0
<<<<<<< HEAD
}
template
void Approximator<float>::add_error( float * const in, int size, int type, float error);
template
void Approximator<double>::add_error( double * const in, int size, int type, double error);

// limit_mag
// limits the magnitude of floating point values to emulate a fixed point representation
// in: data array
// size: number of array elements
// mag: number of bits given for integer part of a number

template <typename Dtype>
void Approximator<Dtype>::limit_mag( Dtype * const in, int size, int mag ) {
  int max_val = 1 << mag;
  for (int i=0; i<size; i++){
    in[i] = (in[i] > max_val)? max_val: in[i];
    in[i] = (in[i] < -max_val)? -max_val: in[i];
  }
}
template 
void Approximator<float>::limit_mag(float * const in, int size, int mag );
template 
void Approximator<double>::limit_mag(double * const in, int size, int mag );

// limit_prec
// limits the precision of floating point values to emulate a fixed point representation
// in: data array
// size: number of array elements
// prec: number of bits given for fractional part of a number

template <typename Dtype>
void Approximator<Dtype>::limit_prec( Dtype * const in, int size, int prec ) {
  for (int i=0; i<size; i++){
    Dtype temp = in[i] * (1<<prec);
    temp = trunc(temp);
    in[i] = temp / (1<<prec);
  }
}
template 
void Approximator<float>::limit_prec(float * const in, int size, int prec );
template 
void Approximator<double>::limit_prec(double * const in, int size, int prec );

template <typename Dtype>
void Approximator<Dtype>::limit_mag_prec( Dtype * const in, int size, int mag, int prec ) {
  Dtype max_val = 1.0*(1 << mag) - (1.0/(1<<(prec)));
  Dtype min_val = -1.0*(1 << mag);
  for (int i=0; i<size; i++){
    Dtype temp = in[i] * (1<<prec);
    temp = round(temp);
    in[i] = temp / (1<<prec);

    in[i] = (in[i] > max_val)? max_val: in[i];
    in[i] = (in[i] < min_val)? min_val: in[i];
  }
}
template 
void Approximator<float>::limit_mag_prec(float * const in, int size, int mag, int prec );
template 
void Approximator<double>::limit_mag_prec(double * const in, int size, int mag, int prec );

=======


}

template
  void add_error<float>( float * const in, int size) ;

template
  void add_error<double>( double * const in, int size) ;
>>>>>>> 7c8c68eca96cc3284f8dd2e7920b6e254063b65e
} // namespace caffe
