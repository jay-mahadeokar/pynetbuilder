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
    from lego.hybrid import FireLego, ConvBNReLULego
    from lego.data import ImageDataLego
    from lego.base import BaseLegoFunction

    netspec = caffe.NetSpec()

    # Data layer
    params = dict(name='data', source='tmp' , batch_size=100, include='test', mean_file='tmp')
    data, label = ImageDataLego(params).attach(netspec)

    # Stage 1
    params = dict(name='1', num_output=96, kernel_size=7,
                  pad=0, stride=2, use_global_stats=True)
    stage1 = ConvBNReLULego(params).attach(netspec, [data])
    params = dict(kernel_size=3, stride=2, pool=P.Pooling.MAX, name='pool1')
    pool1 = BaseLegoFunction('Pooling', params).attach(netspec, [stage1])

    # Fire modules 2 - 9
    num_output = 16
    last = pool1
    ctr = 0
    for i in range(2, 7):
        params = dict(name='fire' + str(i), squeeze_num_output=num_output, use_global_stats=True)
        last = FireLego(params).attach(netspec, [last])
        ctr += 1
        if ctr % 2 == 0:
            num_output += 16

    # Pool8
    pool_params = dict(kernel_size=3, stride=2, pool=P.Pooling.MAX, name='pool8')
    pool = BaseLegoFunction('Pooling', pool_params).attach(netspec, [last])

    # Fire9
    params = dict(name='fire9', squeeze_num_output=num_output, use_global_stats=True)
    fire9 = FireLego(params).attach(netspec, [pool])

    # Conv10
    params = dict(name='conv10', num_output=1000, kernel_size=1, pad=1,
                  stride=1, use_global_stats=True)
    conv10 = ConvBNReLULego(params).attach(netspec, [fire9])

    # pool10
    pool_params = dict(kernel_size=1, stride=1, pool=P.Pooling.AVE, name='pool10')
    pool10 = BaseLegoFunction('Pooling', pool_params).attach(netspec, [conv10])
    smax_loss = BaseLegoFunction('SoftmaxWithLoss', dict(name='loss')).attach(netspec, [pool10, label])

    print netspec.to_proto()










