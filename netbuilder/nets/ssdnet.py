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


'''
    This code is borrowed from here:
    https://github.com/weiliu89/caffe/blob/ssd/python/caffe/model_libs.py
'''
def AddExtraLayers(netspec, last):
    from lego.hybrid import ConvBNLego
    from lego.base import BaseLegoFunction

    use_relu = True
    for i in xrange(6, 9):
        name = 'conv' + str(i) + '_1'
        num_output = 256 if i == 6 else 128
        params = dict(name=name, kernel_size=1, num_output=num_output, pad=0,
                  stride=1)  # use_global_stats=True)
        conv1 = BaseLegoFunction('Convolution', params).attach(netspec, [last])
        relu1 = BaseLegoFunction('ReLU', dict(name=name + '_relu')).attach(netspec, [conv1])

        name = 'conv' + str(i) + '_2'
        num_output = 512 if i == 6 else 256
        params = dict(name=name, kernel_size=3, num_output=num_output, pad=1,
                  stride=2)  # use_global_stats=True)
        conv2 = BaseLegoFunction('Convolution', params).attach(netspec, [relu1])
        last = BaseLegoFunction('ReLU', dict(name=name + '_relu')).attach(netspec, [conv2])

    # Add global pooling layer.
    pool_param = dict(name='pool6', pool=P.Pooling.AVE, global_pooling=True)
    pool = BaseLegoFunction('Pooling', pool_param).attach(netspec, [last])

    return netspec


def get_vgg_ssdnet(is_train=True):

    from lego.ssd import MBoxUnitLego, MBoxAssembleLego
    from lego.base import BaseLegoFunction
    from imagenet import VGGNet
    import math
    netspec = VGGNet().stitch()

    AddExtraLayers(netspec, netspec['fc7'])

    num_classes = 21
    min_dim = 300
    mbox_source_layers = ['conv4_3', 'fc7', 'conv6_2', 'conv7_2', 'conv8_2', 'pool6']
    min_ratio = 20
    max_ratio = 95
    step = int(math.floor((max_ratio - min_ratio) / (len(mbox_source_layers) - 2)))
    min_sizes = []
    max_sizes = []
    for ratio in xrange(min_ratio, max_ratio + 1, step):
        min_sizes.append(min_dim * ratio / 100.)
        max_sizes.append(min_dim * (ratio + step) / 100.)
    min_sizes = [min_dim * 10 / 100.] + min_sizes
    max_sizes = [[]] + max_sizes
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]]
    normalizations = [20, -1, -1, -1, -1, -1]

    assemble_params = dict(mbox_source_layers=mbox_source_layers,
                           normalizations=normalizations,
                           aspect_ratios=aspect_ratios,
                           num_classes=num_classes,
                           min_sizes=min_sizes,
                           max_sizes=max_sizes, is_train=is_train)
    MBoxAssembleLego(assemble_params).attach(netspec, [netspec['label']])

    return netspec


def add_extra_layers_resnet(netspec, last, params):
    from lego.hybrid import ShortcutLego, ConvBNReLULego

    from lego.base import BaseLegoFunction

    blocks = params['extra_blocks']
    num_outputs = params['extra_num_outputs']
    is_train = params['is_train']
    main_branch = params['main_branch']

    use_global_stats = False if is_train else True


    for stage in range(len(blocks)):
        for block in range(blocks[stage]):
            if block == 0:
                shortcut = 'projection'
                stride = 2
            else:
                shortcut = 'identity'
                stride = 1

            name = 'stage' + str(stage + 4) + '_block' + str(block)
            curr_num_output = num_outputs[stage]

            params = dict(name=name, num_output=curr_num_output,
                          shortcut=shortcut, main_branch=main_branch,
                          stride=stride, use_global_stats=use_global_stats,
                          filter_mult=None)
            last = ShortcutLego(params).attach(netspec, [last])

    # Add global pooling layer.
    pool_param = dict(name='pool_last', pool=P.Pooling.AVE, global_pooling=True)
    pool = BaseLegoFunction('Pooling', pool_param).attach(netspec, [last])
    return netspec

def get_resnet_ssdnet(params):
    from lego.ssd import MBoxUnitLego, MBoxAssembleLego
    from lego.base import BaseLegoFunction
    from imagenet import ResNet
    import math


    is_train = params['is_train']
    main_branch = params['main_branch']
    num_output_stage1 = params['num_output_stage1']
    fc_layers = params['fc_layers']
    blocks = params['blocks']
    extra_layer_attach = params['extra_layer_attach']

    netspec = ResNet().stitch(is_train=is_train, source='tt', main_branch=main_branch,
                              num_output_stage1=num_output_stage1, fc_layers=fc_layers, blocks=blocks)

    netspec = add_extra_layers_resnet(netspec, netspec[extra_layer_attach], params)

    num_classes = params['num_classes']
    min_dim = 300
    mbox_source_layers = params['mbox_source_layers']

    min_ratio = 20
    max_ratio = 95
    step = int(math.floor((max_ratio - min_ratio) / (len(mbox_source_layers) - 2)))
    min_sizes = []
    max_sizes = []
    for ratio in xrange(min_ratio, max_ratio + 1, step):
        min_sizes.append(min_dim * ratio / 100.)
        max_sizes.append(min_dim * (ratio + step) / 100.)
    min_sizes = [min_dim * 10 / 100.] + min_sizes
    max_sizes = [[]] + max_sizes
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]]
    # L2 normalize conv4_3.
    normalizations = [20, 20, 20, -1, -1, -1]
    # normalizations = [-1, -1, -1, -1, -1, -1]

    print min_sizes
    print max_sizes
    print aspect_ratios
    print normalizations
    assemble_params = dict(mbox_source_layers=mbox_source_layers,
                           normalizations=normalizations,
                           aspect_ratios=aspect_ratios,
                           num_classes=num_classes,
                           min_sizes=min_sizes,
                           max_sizes=max_sizes, is_train=is_train)
    MBoxAssembleLego(assemble_params).attach(netspec, [netspec['label']])

    return netspec
