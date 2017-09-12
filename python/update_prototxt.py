import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("prototxt"          , help="input net prototxt")
    parser.add_argument("param"             , help="param to set")
    parser.add_argument("prec",  type=int   , help="precision")
    parser.add_argument("scale", type=float , help="scale")
    parser.add_argument("type", type=str , default='conv', help="layer type to set [conv/fc/both]")
    args = parser.parse_args()
    return args

args = parse_args()

param = args.param
prec = args.prec
scale = args.scale

# import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format

# Read net prototxt
netproto = caffe_pb2.NetParameter()
with open(args.prototxt) as f:
    text_format.Merge(str(f.read()), netproto)

typeList = [];
if args.type == 'conv':
    typeList.append('Convolution')
elif args.type == 'fc':
    typeList.append('InnerProduct')
elif args.type == 'both':
    typeList.append('Convolution')
    typeList.append('InnerProduct')
else:
    print 'Error: unknown layer type', args.type
    sys.exit(1)

for layer in netproto.layer:
    if layer.type in typeList:
    # if layer.type in ['Convolution']:
        print 'setting precision', layer.name, param, prec, scale
        # support substring matching
        # param='act' will set both fwd_act and bwd_act
        if param in 'fwd_act':
            layer.fwd_act_precision_param.precision=prec
            layer.fwd_act_precision_param.scale=scale
        if param in 'fwd_wgt':
            layer.fwd_wgt_precision_param.precision=prec
            layer.fwd_wgt_precision_param.scale=scale
        if param in 'bwd_act':
            layer.bwd_act_precision_param.precision=prec
            layer.bwd_act_precision_param.scale=scale
        if param in 'bwd_wgt':
            layer.bwd_wgt_precision_param.precision=prec
            layer.bwd_wgt_precision_param.scale=scale
        if param in 'bwd_grd':
            layer.bwd_grd_precision_param.precision=prec
            layer.bwd_grd_precision_param.scale=scale

# Write the solver to a temporary file and return its filename.
with open(args.prototxt, 'w') as f:
    f.write(str(netproto))
