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


def test_complexity():
    from lego.base import BaseLegoFunction
    from lego.hybrid import ConvBNReLULego
    from tools.complexity import get_complexity

    n = caffe.NetSpec()
    params = dict(name='data', batch_size=16, ntop=2,
                  transform_param=dict(crop_size=224),
                  memory_data_param=dict(batch_size=16, channels=3,
                                         height=224, width=224))
    n.data, n.label = BaseLegoFunction('MemoryData', params).attach(n, [])
    params = dict(name='1', num_output=64, kernel_size=7,
                  use_global_stats=True, pad=3, stride=2)
    stage1 = ConvBNReLULego(params).attach(n, [n.data])
    params = dict(kernel_size=3, stride=2, pool=P.Pooling.MAX, name='pool1')
    pool1 = BaseLegoFunction('Pooling', params).attach(n, [stage1])


    ip_params = dict(name='fc10', num_output=10,
                     param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    ip = BaseLegoFunction('InnerProduct', ip_params).attach(n, [pool1])
    smax_loss = BaseLegoFunction('SoftmaxWithLoss', dict(name='loss')).attach(n, [ip, n.label])

    params, conn = get_complexity(netspec=n)

    print >> sys.stderr, params, conn
