
import caffe
from google.protobuf import text_format
from caffe.proto import caffe_pb2

def read_solver_parameter(filename):
    s = caffe_pb2.SolverParameter()

    with open(filename) as f:
        text_format.Merge(str(f.read()), s)

    return s
