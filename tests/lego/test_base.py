"""
Copyright 2016 Yahoo Inc.
Licensed under the terms of the 2 clause BSD license. 
Please see LICENSE file in the project root for terms.
"""

from caffe.proto import caffe_pb2
import google.protobuf as pb
from caffe import layers as L
from caffe import params as P
import caffe

import sys
sys.path.append('./netbuilder')

def test_conv():
    from lego.base import BaseLegoFunction
    n = caffe.NetSpec()
    n.data, n.label = L.ImageData(image_data_param=dict(source='tmp' , batch_size=100),
                                   ntop=2, transform_param=dict(mean_file='tmp'))
    conv_params = dict(name='conv1', kernel_size=5, num_output=16, pad=2, stride=1)
    conv_lego = BaseLegoFunction('Convolution', conv_params)
    conv_lego.attach(n, [n.data])
    assert n[conv_params['name']] is not None
    print >> sys.stderr, n.to_proto()

def test_relu():
    from lego.base import BaseLegoFunction
    n = caffe.NetSpec()
    n.data, n.label = L.ImageData(image_data_param=dict(source='tmp' , batch_size=100),
                                   ntop=2, transform_param=dict(mean_file='tmp'))
    BaseLegoFunction('ReLU', dict(name='relu1')).attach(n, [n.data])
    assert n['relu1'] is not None
    # print >> sys.stderr, n.to_proto()

def test_config():
    from lego.base import Config
    conv_default = Config.get_default_params('Convolution')
    assert conv_default is not None

    Config.set_default_params('Convolution', 'param', [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    conv_default = Config.get_default_params('Convolution')
    assert conv_default is not None

