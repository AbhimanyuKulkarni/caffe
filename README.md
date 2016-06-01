# Caffe

My branch of Caffe including the set of networks I've been using

Quick setup:

follow instructions here to install caffe: http://caffe.berkeleyvision.org/installation.html
run download_all_models.sh to get pretrained weights for each network
run ./test.sh models/alexnet (for example) to test the C++ interface

save_net.pl is a script to generate data traces using the python interface, see save_net.pl -h

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
