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

parser = ArgumentParser(description=""" This script generates cifar10 resnet train_val.prototxt files""")
parser.add_argument('-n', '--N', help="""Number of block per stage (or N), as described in paper. 
Total number of layers will be 3N + 2""", required=True)
parser.add_argument('-m', '--main_branch', help="""Fire, normal, bottleneck""", required=True)
parser.add_argument('-o', '--output_folder', help="""Train and Test prototxt will be generated as train.prototxt and test.prototxt""")

sys.path.append('netbuilder')


def write_prototxt(is_train, output_folder, N, main_branch):
    from lego.hybrid import ConvBNReLULego, EltwiseReLULego, ShortcutLego
    from lego.base import BaseLegoFunction

    netspec = caffe.NetSpec()

    num_output = 16

    if is_train:
        include = 'train'
        use_global_stats = False
        batch_size = 128
    else:
        include = 'test'
        use_global_stats = True
        batch_size = 1

    # Data layer
    params = dict(name='data', batch_size=1, ntop=2,
                  transform_param=dict(crop_size=32),
                  memory_data_param=dict(batch_size=1, channels=3, height=32, width=32)
                  )
    netspec.data, netspec.label = BaseLegoFunction('MemoryData', params).attach(netspec, [])
    # 1st conv
    params = dict(name='1', num_output=num_output, kernel_size=3,
                  pad=1, stride=1, use_global_stats=use_global_stats)
    conv1 = ConvBNReLULego(params).attach(netspec, [netspec.data])

    last = conv1

    for stage in range(3):
        for block in range(N):

            # subsample at start of every stage except 1st
            if stage > 0 and block == 0:
                shortcut = 'projection'
                stride = 2
            else:
                shortcut = 'identity'
                stride = 1

            name = 'stage' + str(stage) + '_block' + str(block)
            curr_num_output = num_output * (2 ** (stage))

            params = dict(name=name, num_output=curr_num_output,
                          shortcut=shortcut, main_branch=main_branch,
                          stride=stride, use_global_stats=use_global_stats)
            last = ShortcutLego(params).attach(netspec, [last])

    # Last stage
    pool_params = dict(kernel_size=8, stride=1, pool=P.Pooling.AVE, name='pool')
    pool = BaseLegoFunction('Pooling', pool_params).attach(netspec, [last])
    ip_params = dict(name='fc10', num_output=10,
                     param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    ip = BaseLegoFunction('InnerProduct', ip_params).attach(netspec, [pool])
    smax_loss = BaseLegoFunction('SoftmaxWithLoss', dict(name='loss')).attach(netspec, [ip, netspec.label])

    if include == 'test':
        BaseLegoFunction('Accuracy', dict(name='accuracy')).attach(netspec, [ip, netspec.label])
    filename = 'train.prototxt' if is_train else 'test.prototxt'
    filepath = output_folder + '/' + filename
    fp = open(filepath, 'w')
    print >> fp, netspec.to_proto()
    fp.close()



if __name__ == '__main__':
    args = parser.parse_args()
    write_prototxt(True, args.output_folder, int(args.N), args.main_branch)
    write_prototxt(False, args.output_folder, int(args.N), args.main_branch)

    from tools.complexity import get_complexity
    filepath = args.output_folder + '/train.prototxt'
    params, flops = get_complexity(prototxt_file=filepath)
    print 'Number of params: ', (1.0 * params) / 1000000.0, ' Million'
    print 'Number of flops: ', (1.0 * flops) / 1000000.0, ' Million'
