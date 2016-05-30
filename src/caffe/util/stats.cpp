#include <float.h>
#include <stdio.h>
#include <math.h>

#include "caffe/util/stats.hpp"

template <typename Dtype>
StatTracker<Dtype>::StatTracker(int num_bins, int bin_size, bool center) :
  num_bins_(num_bins),
  bin_size_(bin_size),
  center_(center)
{
  histo_ = new int [num_bins_];
  for (int i=0; i<num_bins_; i++){
    histo_[i]=0;
  }
  max_ = -FLT_MAX;
  min_ = FLT_MAX;
  mean_ = 0.0;
  s_ = 0.0;
  count_ = 1;
}
template StatTracker<float>::StatTracker(int num_bins, int bin_size, bool center);
template StatTracker<double>::StatTracker(int num_bins, int bin_size, bool center);
template StatTracker<int>::StatTracker(int num_bins, int bin_size, bool center);

template <typename Dtype>
StatTracker<Dtype>::~StatTracker(){
  delete histo_;
}
template StatTracker<float>::~StatTracker();
template StatTracker<double>::~StatTracker();
template StatTracker<int>::~StatTracker();

template <typename Dtype>
double StatTracker<Dtype>::get_var(){
  switch (count_) {
    case 0: return 0; break;
    case 1: return s_; break;
    default:  return (s_ / (count_ - 1)); break;
  }
}
template double StatTracker<float>::get_var();
template double StatTracker<double>::get_var();
template double StatTracker<int>::get_var();

template <typename Dtype>
double StatTracker<Dtype>::get_stddev(){
  return sqrt(get_var());
}
template double StatTracker<float>::get_stddev();
template double StatTracker<double>::get_stddev();
template double StatTracker<int>::get_stddev();

template <typename Dtype>
int StatTracker<Dtype>::add(const Dtype * data, int count) {
  for (int i=0; i<count; i++){
    max_ = (data[i] > max_)? data[i] : max_;
    min_ = (data[i] < min_)? data[i] : min_;

    int idx = 0;
    if (center_){
      idx = (int)(data[i]/bin_size_ + 0.5) + (num_bins_/2);
    } else {
      idx = (int)(data[i]/bin_size_ + 0.5);
    }
    idx = (idx < 0)? 0 : idx;
    idx = (idx > num_bins_-1)? num_bins_-1 : idx;
    histo_[idx]++;

    update_mean_sd(data[i]);
  }
  return 0;
}
template int StatTracker<float>::add(const float * data, int count);
template int StatTracker<double>::add(const double * data, int count);
template int StatTracker<int>::add(const int * data, int count);


template <typename Dtype>
void StatTracker<Dtype>::update_mean_sd(Dtype val) {
  double tempMean = mean_;
  mean_ += (val - tempMean)/count_;
  s_ += (val - tempMean) * (val - mean_);
  count_++;
}
template void StatTracker<float>::update_mean_sd(float val);
template void StatTracker<double>::update_mean_sd(double val);
template void StatTracker<int>::update_mean_sd(int val);

template <>
void StatTracker<float>::print(char const* name){
  printf("stats: %s, %lld, %f, %f, %f, %f\n", name, count_, min_, max_, mean_, get_stddev());
}
template <>
void StatTracker<double>::print(char const* name){
  printf("stats: %s, %lld, %f, %f, %f, %f\n", name, count_, min_, max_, mean_, get_stddev());
}
template <>
void StatTracker<int>::print(char const* name){
  printf("stats: %s, %lld, %d, %d, %f, %f\n", name, count_, min_, max_, mean_, get_stddev());
}

template <typename Dtype>
void StatTracker<Dtype>::print_histo(char const* name){
  printf("%s,", name);
  for (int i=0; i<num_bins_; i++){
    if (center_){
      printf("%d,", bin_size_ * (i-(num_bins_/2)));
    } else {
      printf("%d,", i);
    }
  }
  printf("\n");
  printf("%s,", name);
  for (int i=0; i<num_bins_; i++){
    printf("%d,", histo_[i]);
  }
  printf("\n");
}
template void StatTracker<float>::print_histo(char const* name);
template void StatTracker<double>::print_histo(char const* name);
template void StatTracker<int>::print_histo(char const* name);

