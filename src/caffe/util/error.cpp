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

namespace caffe {

enum eERROR_TYPE {
  NORMAL = 0,
  ADD_LOGNORMAL = 1,
  SUB_LOGNORMAL = 2,
  BITFLIP_SEF = 3,
  BITFLIP_EF = 4,
  BITFLIP_F = 5,
};

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

    int errorSize = size; // size of error matrix should match output of gemm
    

    switch (error_type) {
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
          switch (error_type) {
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
              printf("Unsupported error_type %d\n", error_type);
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

          // initialize poisson RNG
          Dtype lambda = 1/error - 1;
          CHECK_GT(lambda, 0);
          boost::poisson_distribution<int> poisson_dist(lambda);
          boost::variate_generator<caffe::rng_t*, boost::poisson_distribution<int> >
            poisson_vg(caffe_rng(), poisson_dist);

          // initialize uniform RNG
          int num_bits;
          switch (error_type) {
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
        printf("Unknown error type: %d\n", error_type);
        assert(false);
        break;
    } // switch error_type
  } //if error > 0


}

template
  void add_error<float>( float * const in, int size) ;

template
  void add_error<double>( double * const in, int size) ;
} // namespace caffe
