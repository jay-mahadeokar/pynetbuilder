"""
Copyright 2016 Yahoo Inc.
Licensed under the terms of the 2 clause BSD license. 
Please see LICENSE file in the project root for terms.
"""

import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format
caffe.set_mode_cpu()
import tempfile
import os
import sys

'''
    Simple handle to create netspec files. This code snippet is on lines
    of: https://github.com/BVLC/caffe/blob/master/python/caffe/test/test_net.py
'''
def _create_file_from_netspec(netspec):
    f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    f.write(str(netspec.to_proto()))
    return f.name


'''
    This is a utility function which computes the 
    complexity of a given network.
    This is slightly modified version from the one written by @fsachin, 
    for his network profile ipython notebook
'''
def get_complexity(netspec=None, prototxt_file=None):
    # One of netspec, or prototxt_path params should not be None
    assert (netspec is not None) or (prototxt_file is not None)

    if netspec is not None:
        prototxt_file = _create_file_from_netspec(netspec)
    net = caffe.Net(prototxt_file, caffe.TEST)

    total_params = 0
    total_flops = 0

    net_params = caffe_pb2.NetParameter()
    text_format.Merge(open(prototxt_file).read(), net_params)

    for layer in net_params.layer:
        if layer.name in net.params:

            params = net.params[layer.name][0].data.size
            # If convolution layer, multiply flops with receptive field
            # i.e. #params * datawidth * dataheight
            if layer.type == 'Convolution':  # 'conv' in layer:
                data_width = net.blobs[layer.name].data.shape[2]
                data_height = net.blobs[layer.name].data.shape[3]
                flops = net.params[layer.name][0].data.size * data_width * data_height
                # print >> sys.stderr, layer.name, params, flops
            else:
                flops = net.params[layer.name][0].data.size

            total_params += params
            total_flops += flops



    if netspec is not None:
        os.remove(prototxt_file)

    return total_params, total_flops

