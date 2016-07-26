from caffe.proto import caffe_pb2
import google.protobuf as pb
from caffe import layers as L
from caffe import params as P
import caffe

import sys
sys.path.append('../netbuilder')

def test_conv_bn_relu_lego():
    from lego.hybrid import ConvBNReLULego
    n = caffe.NetSpec()
    n.data, n.label = L.ImageData(image_data_param=dict(source='tmp' , batch_size=100),
                                   ntop=2, transform_param=dict(mean_file='tmp'))
    params = dict(name='1', kernel_size=5, num_output=16, pad=2,
                   stride=1, use_global_stats=False)
    lego = ConvBNReLULego(params)
    lego.attach(n, [n.data])
    assert n['conv_' + params['name']] is not None
    print >> sys.stderr, n.to_proto()

def test_conv_bn_lego():
    from lego.hybrid import ConvBNLego
    n = caffe.NetSpec()
    n.data, n.label = L.ImageData(image_data_param=dict(source='tmp' , batch_size=100),
                                   ntop=2, transform_param=dict(mean_file='tmp'))
    params = dict(name='1', kernel_size=5, num_output=16, pad=2,
                  stride=1, use_global_stats=True)
    lego = ConvBNLego(params)
    lego.attach(n, [n.data])
    assert n['conv_' + params['name']] is not None
    # print >> sys.stderr, n.to_proto()

def test_eltwise_relu_lego():
    from lego.hybrid import EltwiseReLULego
    n = caffe.NetSpec()
    n.data, n.label = L.ImageData(image_data_param=dict(source='tmp' , batch_size=100),
                                   ntop=2, transform_param=dict(mean_file='tmp'))
    EltwiseReLULego(dict(name='1')).attach(n, [n.data, n.label])
    assert n['eltwise_1'] is not None
    # print >> sys.stderr, n.to_proto()

def test_fire_lego():
    from lego.hybrid import FireLego
    n = caffe.NetSpec()
    n.data, n.label = L.ImageData(image_data_param=dict(source='tmp' , batch_size=100),
                                   ntop=2, transform_param=dict(mean_file='tmp'))
    params = dict(name='fire1', squeeze_num_output=16, use_global_stats=True)
    FireLego(params).attach(n, [n.data])
    # print >> sys.stderr, n.to_proto()


def test_inception_v1_lego():
    from lego.hybrid import InceptionV1Lego
    n = caffe.NetSpec()
    n.data, n.label = L.ImageData(image_data_param=dict(source='tmp' , batch_size=100),
                                   ntop=2, transform_param=dict(mean_file='tmp'))
    params = dict(name='inception1', num_outputs=[16, 96, 128, 16, 32, 32], use_global_stats=True)
    InceptionV1Lego(params).attach(n, [n.data])
    # print >> sys.stderr, n.to_proto()

def test_shortcut_lego():
    from lego.hybrid import ShortcutLego
    n = caffe.NetSpec()
    n.data, n.label = L.ImageData(image_data_param=dict(source='tmp' , batch_size=100),
                                   ntop=2, transform_param=dict(mean_file='tmp'))
    params = dict(name='block1', shortcut='projection', num_output=64, main_branch='inception', stride=1, use_global_stats=True)
    ShortcutLego(params).attach(n, [n.data])
    # print >> sys.stderr, n.to_proto()

