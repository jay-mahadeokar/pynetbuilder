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

from lego.hybrid import ConvReLULego
from lego.base import BaseLegoFunction, BaseLego, Config
from tools.complexity import get_complexity

'''
    Base class defining the stitch
    functionality
'''
class BaseNet():
    def __init__(self):
        pass

    def stitch(self, **kwargs):
        pass


'''
    Class to stitch together a residual network
'''
class ResNet(BaseNet):

    def stitch(self, is_train, source, main_branch, num_output_stage1, fc_layers, blocks):
        from lego.hybrid import ConvBNReLULego, EltwiseReLULego, ShortcutLego, ConvBNLego
        from lego.data import ImageDataLego
        from lego.base import BaseLegoFunction

        netspec = caffe.NetSpec()

        if is_train:
            include = 'train'
            use_global_stats = False
            batch_size = 256
        else:
            include = 'test'
            use_global_stats = True
            batch_size = 1

        # Freeze 1st 2 stages and dont update batch norm stats
        Config.set_default_params('Convolution', 'param', [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
        if is_train:
            use_global_stats = True


        # Data layer, its dummy, you need to replace this with Annotated data layer
        params = dict(name='data', batch_size=1, ntop=2,
                      memory_data_param=dict(batch_size=1, channels=3, height=300, width=300)
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
                if block == 0 and stage == 0 and main_branch == 'normal':
                    shortcut = 'identity'

                # This is for not downsampling while creating detection
                # network
                # if block == 0 and stage == 1:
                #    stride = 1

                name = 'stage' + str(stage) + '_block' + str(block)
                curr_num_output = num_output * (2 ** (stage))

                params = dict(name=name, num_output=curr_num_output,
                              shortcut=shortcut, main_branch=main_branch,
                              stride=stride, use_global_stats=use_global_stats,)
                last = ShortcutLego(params).attach(netspec, [last])

            # TODO: Note this should be configurable
            if stage == 0:
               Config.set_default_params('Convolution', 'param', [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
               if is_train:
                   use_global_stats = False


        '''
                You should modify these layers in order to experiment with different 
                architectures specific for detection
        '''
        if not fc_layers:
            # Last stage
            pool_params = dict(kernel_size=7, stride=1, pool=P.Pooling.AVE, name='pool', pad=3)
            pool = BaseLegoFunction('Pooling', pool_params).attach(netspec, [last])
        else:

            '''pool_params = dict(name='pool_before1024', kernel_size=3, stride=1, pool=P.Pooling.MAX, pad=1)
            pool = BaseLegoFunction('Pooling', pool_params).attach(netspec, [last])
            conv_last1_params = dict(name='3by3_1024', num_output=1024, use_global_stats=use_global_stats,
                     # pad=0, kernel_size=1)
                     pad=1, kernel_size=3, dilation=3)
            conv_last1 = ConvBNReLULego(conv_last1_params).attach(netspec, [pool])
            '''

            conv_last1_params = dict(name='1by1_2048', num_output=2048, use_global_stats=use_global_stats,
                                 pad=1, kernel_size=1, dilation=1)
                                 # pad=3, kernel_size=7)
            conv_last1 = ConvBNReLULego(conv_last1_params).attach(netspec, [last])

            pool_last_params = dict(kernel_size=7, stride=1, pool=P.Pooling.AVE, name='pool', pad=3)
            pool_last = BaseLegoFunction('Pooling', pool_last_params).attach(netspec, [conv_last1])

            conv_last2_params = dict(name='1by1_4096', kernel_size=1, num_output=4096, use_global_stats=use_global_stats,
                                     stride=1, pad=0)
            conv_last2 = ConvBNReLULego(conv_last2_params).attach(netspec, [pool_last])

        return netspec

'''
    Class to stitch together VGGNet
    Some code borrowed from 
    https://github.com/weiliu89/caffe/blob/ssd/python/caffe/model_libs.py
'''
class VGGNet(BaseNet):

    def stitch(self, is_train=True, reduced=True, fully_conv=True, dilated=True, dropout=False):
        netspec = caffe.NetSpec()
        if is_train:
            include = 'train'
            use_global_stats = False
            batch_size = 128
        else:
            include = 'test'
            use_global_stats = True
            batch_size = 1

        Config.set_default_params('Convolution', 'param', [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])

        # Data layer
        params = dict(name='data', batch_size=1, ntop=2,
                      transform_param=dict(crop_size=300),
                      memory_data_param=dict(batch_size=1, channels=3, height=300, width=300)
                      )
        netspec.data, netspec.label = BaseLegoFunction('MemoryData', params).attach(netspec, [])

        last = netspec.data
        num_outputs = [64, 128, 256, 512, 512]
        # Conv layers stages
        for stage in range(1, 6):
            blocks = 2 if stage < 3 else 3
            for b in range(1, blocks + 1):
                name = str(stage) + '_' + str(b)

                params = dict(name=name, num_output=num_outputs[stage - 1], kernel_size=3, pad=1, stride=1)
                last = ConvReLULego(params).attach(netspec, [last])

            if dilated and stage == 5:
                pool_params = dict(name='pool' + str(stage), kernel_size=3, stride=1, pool=P.Pooling.MAX, pad=1)
            else:
                pool_params = dict(name='pool' + str(stage), kernel_size=2, stride=2, pool=P.Pooling.MAX)
            last = BaseLegoFunction('Pooling', pool_params).attach(netspec, [last])

            if stage == 2:
                Config.set_default_params('Convolution', 'param', [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])


        # FC layers
        if fully_conv:
            # FC6 fully convolutional
            if dilated:
                if reduced:
                    ip_params = dict(name='fc6', num_output=1024, pad=6, kernel_size=3, dilation=6)
                else:
                    ip_params = dict(name='fc6', num_output=4096, pad=6, kernel_size=7, dilation=2)
            else:
                if reduced:
                    ip_params = dict(name='fc6', num_output=1024, pad=3, kernel_size=3, dilation=3)
                else:
                    ip_params = dict(name='fc6', num_output=4096, pad=3, kernel_size=7)

            fc6 = BaseLegoFunction('Convolution', ip_params).attach(netspec, [last])
            relu_params = dict(name='relu6', in_place=True)
            out6 = BaseLegoFunction('ReLU', relu_params).attach(netspec, [fc6])
            if dropout:
                drop_params = dict(name='drop6', dropout_param=dict(dropout_ratio=0.5), in_place=True)
                out6 = BaseLegoFunction('Dropout', drop_params).attach(netspec, [out6])

            # FC7 fully convolutional
            if reduced:
                ip_params = dict(name='fc7', num_output=1024, kernel_size=1)
            else:
                ip_params = dict(name='fc7', num_output=4096, kernel_size=1)
            fc7 = BaseLegoFunction('Convolution', ip_params).attach(netspec, [out6])
            relu_params = dict(name='relu7', in_place=True)
            out7 = BaseLegoFunction('ReLU', relu_params).attach(netspec, [fc7])
            if dropout:
                drop_params = dict(name='drop6', dropout_param=dict(dropout_ratio=0.5), in_place=True)
                out7 = BaseLegoFunction('Dropout', drop_params).attach(netspec, [out7])

        else:
            ip_params = dict(name='fc6', num_output=4096)
            fc6 = BaseLegoFunction('InnerProduct', ip_params).attach(netspec, [last])
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


        return netspec
