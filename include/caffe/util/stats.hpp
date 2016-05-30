#ifndef _STATS_H_
#define _STATS_H_

#include <string>

template <typename Dtype>
class StatTracker {
  private:
    Dtype max_;
    Dtype min_;
    int num_bins_;
    int bin_size_;
    bool center_; // center histogram around zero
    int * histo_;
    double mean_;
    Dtype s_; // running variance; std_dev = sqrt(s/(count-2));
    long long count_;
    
    void update_mean_sd(Dtype val);
  public:
    StatTracker(int num_bins, int bin_size, bool center);
    ~StatTracker();
    int add(const Dtype * data, int count);
    double get_var();
    double get_stddev();
    void print(char const* name);
    void print_histo(char const* name);
};

#endif
