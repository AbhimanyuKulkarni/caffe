#ifndef _STATS_H_
#define _STATS_H_

#include <string>

#define GEN_HISTO 0
#define NUM_BINS 200
#define BIN_SIZE 10

template <typename Dtype>
class StatTracker {
  private:
    Dtype max_;
    Dtype min_;
    int histo_[NUM_BINS];
    Dtype mean_;
    Dtype s_; // running variance; std_dev = sqrt(s/(count-2));
    long long count_;
    
    void update_mean_sd(Dtype val);
  public:
    StatTracker();
//    ~StatTracker();
    int add(const Dtype * data, int count);
    Dtype get_var();
    Dtype get_stddev();
    void print(char const* name);
    void print_histo(char const* name);
};

#endif
