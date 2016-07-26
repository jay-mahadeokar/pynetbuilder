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

parser = ArgumentParser(description=""" This script generates imagenet resnet train_val.prototxt files""")
parser.add_argument('-m', '--main_branch', help="""normal, bottleneck""", required=True)
parser.add_argument('-n', '--num_output_stage1', help="""Number of filters in stage 1 of resnet""", type=int, default=128)
parser.add_argument('-o', '--output_folder', help="""Train and Test prototxt will be generated as train.prototxt and test.prototxt""")
parser.add_argument('-b', '--blocks', type=int, nargs='+', help="""Number of Blocks in the 4 resnet stages""", default=[3, 4, 6, 3])
parser.add_argument('--fc_layers', dest='fc_layers', action='store_true')
parser.add_argument('--no-fc_layers', dest='fc_layers', action='store_false')
parser.set_defaults(fc_layers=False)


sys.path.append('netbuilder')

def write_prototxt(is_train, source, output_folder, main_branch, num_output_stage1, fc_layers, blocks):

    from lego.hybrid import ConvBNReLULego, EltwiseReLULego, ShortcutLego
    from lego.data import ImageDataLego
    from lego.base import BaseLegoFunction

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
                  transform_param=dict(crop_size=224),
                  memory_data_param=dict(batch_size=1, channels=3, height=224, width=224)
                  )
    netspec.data, netspec.label = BaseLegoFunction('MemoryData', params).attach(netspec, [])

    # Stage 1
    params = dict(name='1', num_output=64, kernel_size=7,
                  use_global_stats=use_global_stats, pad=3, stride=2)
    stage1 = ConvBNReLULego(params).attach(netspec, [netspec.data])
    params = dict(kernel_size=3, stride=2, pool=P.Pooling.MAX, name='pool1')
    pool1 = BaseLegoFunction('Pooling', params).attach(netspec, [stage1])

    num_output = num_output_stage1

    # Stages 2 - 5
    last = pool1
    for stage in range(4):
        name = str(stage + 1)

        for block in range(blocks[stage]):
            if block == 0:
                shortcut = 'projection'
                if stage > 0:
                    stride = 2
                else:
                    stride = 1
            else:
                shortcut = 'identity'
                stride = 1

            # this is for resnet 18 / 34, where the first block of stage
            # 0 does not need projection shortcut
            if block == 0 and stage == 0 and main_branch in ['normal', 'inception_trick']:
                shortcut = 'identity'


            name = 'stage' + str(stage) + '_block' + str(block)
            curr_num_output = num_output * (2 ** (stage))

            params = dict(name=name, num_output=curr_num_output,
                          shortcut=shortcut, main_branch=main_branch,
                          stride=stride, use_global_stats=use_global_stats,)
            last = ShortcutLego(params).attach(netspec, [last])

    # Last stage
    if not fc_layers:
        pool_params = dict(kernel_size=7, stride=1, pool=P.Pooling.AVE, name='pool')
        pool = BaseLegoFunction('Pooling', pool_params).attach(netspec, [last])
        ip_params = dict(name='fc1000', num_output=1000)
        ip = BaseLegoFunction('InnerProduct', ip_params).attach(netspec, [pool])
    else:
        conv_last1_params = dict(name='1by1_2048', kernel_size=1, num_output=2048, use_global_stats=use_global_stats,
                                 pad=0, stride=1)
        conv_last1 = ConvBNReLULego(conv_last1_params).attach(netspec, [last])

        pool_last_params = dict(kernel_size=7, stride=1, pool=P.Pooling.AVE, name='pool')
        pool_last = BaseLegoFunction('Pooling', pool_last_params).attach(netspec, [conv_last1])
        # pool_last = BaseLegoFunction('Pooling', pool_last_params).attach(netspec, [last])

        conv_last2_params = dict(name='1by1_4096', kernel_size=1, num_output=4096, use_global_stats=use_global_stats,
                                 pad=0, stride=1)
        conv_last2 = ConvBNReLULego(conv_last2_params).attach(netspec, [pool_last])
        drop_last2 = BaseLegoFunction('Dropout', dict(name='drop_1by1_4096', dropout_ratio=0.2)).attach(netspec, [conv_last2])

        conv_last3_params = dict(name='1by1_1000', kernel_size=1, num_output=1000, use_global_stats=use_global_stats,
                                 pad=0, stride=1)
        ip = ConvBNReLULego(conv_last3_params).attach(netspec, [drop_last2])

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
    write_prototxt(True, 'train', args.output_folder, args.main_branch, args.num_output_stage1, args.fc_layers, args.blocks)
    write_prototxt(False, 'test', args.output_folder, args.main_branch, args.num_output_stage1, args.fc_layers, args.blocks)

    from tools.complexity import get_complexity
    filepath = args.output_folder + '/train.prototxt'
    params, flops = get_complexity(prototxt_file=filepath)
    print 'Number of params: ', (1.0 * params) / 1000000.0, ' Million'
    print 'Number of flops: ', (1.0 * flops) / 1000000.0, ' Million'
    print args.fc_layers
