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
from argparse import ArgumentParser

sys.path.append('netbuilder')

if __name__ == '__main__':
    from lego.hybrid import ConvBNReLULego, EltwiseReLULego, ResnetLego
    from lego.data import ImageDataLego
    # from lego.core import PoolLego, InnerProductLego, SoftmaxWithLossLego
    from lego.base import BaseLegoFunction

    netspec = caffe.NetSpec()

    N = 2
    num_output = 16

    # Data layer
    params = dict(name='data', source='/projects/flickr_sciences/jay/cifar10/dataframes/train' , batch_size=64, include='test', mean_file='mean.binaryproto')
    data, label = ImageDataLego(params).attach(netspec)
    params = dict(name='data', source='/projects/flickr_sciences/jay/cifar10/dataframes/train' , batch_size=64, include='train', mean_file='mean.binaryproto')
    data, label = ImageDataLego(params).attach(netspec)

    # 1st conv
    params = dict(name='1', num_output=16, kernel_size=3, pad=1, stride=1)
    conv1 = ConvBNReLULego(params).attach(netspec, [data])

    last = conv1
    for stage in range(3):
        for block in range(2 * N):

            # subsample at start of every stage except 1st
            if stage > 0 and block == 0:
                stride = 2
            else:
                stride = 1
            name = 'stage' + str(stage) + '_block' + str(block)
            curr_num_output = num_output * (2 ** (stage))
            params = dict(name=name + str(), num_output=curr_num_output, kernel_size=3, pad=1, stride=stride)
            last = ConvBNReLULego(params).attach(netspec, [last])

    # Last stage
    pool_params = dict(kernel_size=8, stride=1, pool=P.Pooling.AVE, name='pool')
    pool = BaseLegoFunction('Pooling', pool_params).attach(netspec, [last])

    ip_params = dict(name='fc10', num_output=10,
                     param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    ip = BaseLegoFunction('InnerProduct', ip_params).attach(netspec, [pool])
    smax_loss = BaseLegoFunction('SoftmaxWithLoss', dict(name='loss')).attach(netspec, [ip, label])

    print netspec.to_proto()
