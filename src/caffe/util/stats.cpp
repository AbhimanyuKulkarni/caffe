#include <float.h>
#include <stdio.h>
#include <math.h>

#include "caffe/util/stats.hpp"

template <typename Dtype>
StatTracker<Dtype>::StatTracker(){
  for (int i=0; i<NUM_BINS; i++){
    histo_[i]=0;
  }
  max_ = -FLT_MAX;
  min_ = FLT_MAX;
  mean_ = 0.0;
  s_ = 0.0;
  count_ = 1;
}
template StatTracker<float>::StatTracker();
template StatTracker<double>::StatTracker();

//template <typename Dtype>
//StatTracker<Dtype>::~StatTracker(){
//}
//template StatTracker<float>::~StatTracker();

template <typename Dtype>
Dtype StatTracker<Dtype>::get_var(){
  switch (count_) {
    case 0: return 0; break;
    case 1: return s_; break;
    default:  return (s_ / (count_ - 1)); break;
  }
}
template  float StatTracker<float>::get_var();
template double StatTracker<double>::get_var();

template <typename Dtype>
Dtype StatTracker<Dtype>::get_stddev(){
  return sqrt(get_var());
}
template  float StatTracker<float>::get_stddev();
template double StatTracker<double>::get_stddev();

template <typename Dtype>
int StatTracker<Dtype>::add(const Dtype * data, int count) {
  for (int i=0; i<count; i++){
    max_ = (data[i] > max_)? data[i] : max_;
    min_ = (data[i] < min_)? data[i] : min_;

    int idx = (int)(data[i]/BIN_SIZE + 0.5) + (NUM_BINS/2);
    idx = (idx < 0)? 0 : idx;
    idx = (idx > NUM_BINS-1)? NUM_BINS-1 : idx;
    histo_[idx]++;

    update_mean_sd(data[i]);
  }
  return 0;
}
template int StatTracker<float>::add(const float * data, int count);
template int StatTracker<double>::add(const double * data, int count);


template <typename Dtype>
void StatTracker<Dtype>::update_mean_sd(Dtype val) {
  Dtype tempMean = mean_;
  mean_ += (val - tempMean)/count_;
  s_ += (val - tempMean) * (val - mean_);
  count_++;
}
template void StatTracker<float>::update_mean_sd(float val);
template void StatTracker<double>::update_mean_sd(double val);

template <typename Dtype>
void StatTracker<Dtype>::print(char const* name){
  printf("stats: %s, %lld, %f, %f, %f, %f\n", name, count_, min_, max_, mean_, get_stddev());
}
template void StatTracker<float>::print(char const* name);
template void StatTracker<double>::print(char const* name);

template <typename Dtype>
void StatTracker<Dtype>::print_histo(char const* name){
  for (int i=0; i<NUM_BINS; i++){
    printf("%d,", BIN_SIZE * (i-(NUM_BINS/2)));
  }
  printf("\n");
  for (int i=0; i<NUM_BINS; i++){
    printf("%d,", histo_[i]);
  }
  printf("\n");
}
template void StatTracker<float>::print_histo(char const* name);
template void StatTracker<double>::print_histo(char const* name);

