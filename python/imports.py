import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from google.protobuf import text_format
import caffe
from caffe.proto import caffe_pb2
from caffe import layers as L, params as P
