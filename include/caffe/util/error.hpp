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
    StatTracker<int> * zeros_in_chunks_;
    long long zero_count;
    long long data_count;
  public: 

    Approximator();
    ~Approximator();
    void init(int chunkSize);
    void pre_layer (
      const vector<Blob<Dtype>*> & bottom, 
      const vector<Blob<Dtype>*> & top,
      vector<shared_ptr<Blob<Dtype> > > & weights,
      LayerParameter& param
      );

    void print_stats( const char * name , LayerParameter & param);

    void add_error( Dtype * const in, int size, int type, Dtype error);

    void limit_mag( Dtype * const in, int size, int mag );

    void limit_prec( Dtype * const in, int size, int prec );

    void limit_mag_prec( Dtype * const in, int size, int mag, int prec );

    long long count_zeros( Dtype * const in, int size );

    void zero_out( Dtype * const in, int size, Dtype threshold );

}; // class Approximator

} // namespace caffe
