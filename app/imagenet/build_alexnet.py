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
from lego.hybrid import ConvBNReLULego
from lego.base import BaseLegoFunction
from tools.complexity import get_complexity

parser = ArgumentParser(description=""" This script generates imagenet alexnet train_val.prototxt files""")
parser.add_argument('-o', '--output_folder', help="""Train and Test prototxt will be generated as train.prototxt and test.prototxt""")


def write_prototxt(is_train, source, output_folder):
    netspec = caffe.NetSpec()
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
                  transform_param=dict(crop_size=256),
                  memory_data_param=dict(batch_size=1, channels=3, height=227, width=227)
                  )
    netspec.data, netspec.label = BaseLegoFunction('MemoryData', params).attach(netspec, [])

    # Conv 1
    params = dict(name='1', num_output=96, kernel_size=11, pad=1, stride=4, use_global_stats=use_global_stats)
    conv1 = ConvBNReLULego(params).attach(netspec, [netspec.data])
    pool_params = dict(name='pool_1' , kernel_size=3, stride=2, pool=P.Pooling.MAX)
    pool1 = BaseLegoFunction('Pooling', pool_params).attach(netspec, [conv1])

    # Conv 2
    params = dict(name='2', num_output=256, kernel_size=5, pad=2, stride=1, use_global_stats=use_global_stats)
    conv2 = ConvBNReLULego(params).attach(netspec, [pool1])
    pool_params = dict(name='pool_2' , kernel_size=3, stride=2, pool=P.Pooling.MAX)
    pool2 = BaseLegoFunction('Pooling', pool_params).attach(netspec, [conv2])

    # Conv 3
    params = dict(name='3', num_output=384, kernel_size=3, pad=1, stride=1, use_global_stats=use_global_stats)
    conv3 = ConvBNReLULego(params).attach(netspec, [pool2])

    # Conv 4
    params = dict(name='4', num_output=384, kernel_size=3, pad=1, stride=1, use_global_stats=use_global_stats)
    conv4 = ConvBNReLULego(params).attach(netspec, [conv3])

    # Conv 5
    params = dict(name='5', num_output=256, kernel_size=3, pad=1, stride=1, use_global_stats=use_global_stats)
    conv5 = ConvBNReLULego(params).attach(netspec, [conv4])

    # FC layers
    ip_params = dict(name='fc6', num_output=4096)
    fc6 = BaseLegoFunction('InnerProduct', ip_params).attach(netspec, [conv5])
    relu_params = dict(name='relu6')
    relu6 = BaseLegoFunction('ReLU', relu_params).attach(netspec, [fc6])
    drop_params = dict(name='drop6', dropout_param=dict(dropout_ratio=0.5))
    drop6 = BaseLegoFunction('Dropout', drop_params).attach(netspec, [relu6])

    ip_params = dict(name='fc7', num_output=4096)
    fc7 = BaseLegoFunction('InnerProduct', ip_params).attach(netspec, [drop6])
    relu_params = dict(name='relu7')
    relu7 = BaseLegoFunction('ReLU', relu_params).attach(netspec, [fc7])
    drop_params = dict(name='drop7', dropout_param=dict(dropout_ratio=0.5))
    drop7 = BaseLegoFunction('Dropout', drop_params).attach(netspec, [relu7])

    ip_params = dict(name='fc8', num_output=1000)
    fc8 = BaseLegoFunction('InnerProduct', ip_params).attach(netspec, [drop7])

    # Loss and accuracy
    smax_loss = BaseLegoFunction('SoftmaxWithLoss', dict(name='loss')).attach(netspec, [fc8, netspec.label])

    if include == 'test':
        BaseLegoFunction('Accuracy', dict(name='accuracy')).attach(netspec, [fc8, netspec.label])
    filename = 'train.prototxt' if is_train else 'test.prototxt'
    filepath = output_folder + '/' + filename
    fp = open(filepath, 'w')
    print >> fp, netspec.to_proto()
    fp.close()


if __name__ == '__main__':
    args = parser.parse_args()
    write_prototxt(True, 'train', args.output_folder)
    write_prototxt(False, 'test', args.output_folder)

    # Also print out the network complexity
    filepath = args.output_folder + '/train.prototxt'
    params, flops = get_complexity(prototxt_file=filepath)
    print 'Number of params: ', (1.0 * params) / 1000000.0, ' Million'
    print 'Number of flops: ', (1.0 * flops) / 1000000.0, ' Million'
