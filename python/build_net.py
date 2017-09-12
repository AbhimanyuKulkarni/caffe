# This script computes the blob sizes of a caffe model without having to load the network in caffe

import argparse
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format

def read_net_prototxt(filename):
    netproto = caffe_pb2.NetParameter()
    with open(filename) as f:
        text_format.Merge(str(f.read()), netproto)
    return netproto

def parse_args():
    parser = argparse.ArgumentParser(description="computes blob sizes given a caffe prototxt")
    parser.add_argument( "net", help="network prototxt")
    parser.add_argument( "--phase", help="network phase [TRAIN/TEST]", default="TEST")
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()

    proto = read_net_prototxt(args.net)

    shapemap = dict()

    for layer in proto.layer:
        top_shape = ()
        wgt_shape = ()
        # print '%s (%s)' % (layer.name,layer.type)
        if layer.bottom:
            bottom = layer.bottom[0]

            if bottom in shapemap:
                if 'top' in shapemap[bottom]:
                    bottom_shape = shapemap[bottom]['top']
                else: 
                    raise KeyError, 'no top blob in shapemap[%s]' % bottom
                shapemap[layer.name] = dict()
                shapemap[layer.name]['bottom'] = bottom_shape
            else:
                print bottom,'not in shapemap'

        if layer.type == 'Data':
            # layer.include[0].phase is an enum, comparing strings is more intuitive
            if layer.include and args.phase in str(layer.include[0]):
                n = layer.data_param.batch_size
                c = 3 # assuming 3 channel (RGB) images as input
                h = layer.transform_param.crop_size
                w = layer.transform_param.crop_size
                top_shape = (n,c,h,w)
        if layer.type == 'Convolution':
            (bn,bc,bh,bw) = bottom_shape
            pad = layer.convolution_param.pad[0] if layer.convolution_param.pad else 0
            ks  = layer.convolution_param.kernel_size[0]
            s   = layer.convolution_param.stride[0] if layer.convolution_param.stride else 1
            k   = layer.convolution_param.num_output
            n = bn
            c = k # return a long
            h = (bh + 2 * pad - ks) / s + 1
            w = (bw + 2 * pad - ks) / s + 1
            top_shape = (n,c,h,w)
            wgt_shape = (k,c,ks,ks)
        if layer.type == 'ReLU':
            top_shape = bottom_shape
        if layer.type == 'Pooling':
            (bn,bc,bh,bw) = bottom_shape
            pad = layer.pooling_param.pad[0] if layer.pooling_param.pad else 0
            ks = layer.pooling_param.kernel_size
            s = layer.pooling_param.stride
            n = bn
            c = bc # return a long
            h = (bh + 2 * pad - ks) / s + 1
            w = (bw + 2 * pad - ks) / s + 1
            top_shape = (n,c,h,w)
        if layer.type == 'LRN':
            top_shape = bottom_shape
        if layer.type == 'InnerProduct':
            (bn,bc,bh,bw) = bottom_shape
            k = layer.inner_product_param.num_output
            n = bn
            c = k # return a long
            h = 1
            w = 1
            ks = 1
            top_shape = (n,c,h,w)
            wgt_shape = (k,c,ks,ks)
             
        # add shapes to map
        if top_shape:
            if layer.name not in shapemap:
                shapemap[layer.name] = dict()
            shapemap[layer.name]['top'] = top_shape
        if wgt_shape:
            shapemap[layer.name]['wgt'] = wgt_shape

    # print map at the end
    for layer in proto.layer:
        l = layer.name
        if l in shapemap:
            print l, shapemap[l]
