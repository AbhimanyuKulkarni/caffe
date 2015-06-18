#include "caffe/util/stats.hpp"
#include "caffe/blob.hpp"
#include "caffe/proto/caffe.pb.h"
#include <vector>

namespace caffe {

// Assuming we will only ever
  
template <typename Dtype>
class Approximator {
  private:
    StatTracker<Dtype> stats_;
  public: 
    void pre_layer (
      const vector<Blob<Dtype>*> & bottom, 
      const vector<Blob<Dtype>*> & top,
      vector<shared_ptr<Blob<Dtype> > > & weights,
      LayerParameter& param
      );

    void print_stats(string name);

    void add_error( Dtype * const in, int size, int type, Dtype error);

    void limit_mag( Dtype * const in, int size, int mag );

    void limit_prec( Dtype * const in, int size, int prec );

    void limit_mag_prec( Dtype * const in, int size, int mag, int prec );

}; // class Approximator

} // namespace caffe
