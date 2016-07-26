"""
Copyright 2016 Yahoo Inc.
Licensed under the terms of the 2 clause BSD license. 
Please see LICENSE file in the project root for terms.
"""

from base import BaseLego
from base import BaseLegoFunction

from caffe.proto import caffe_pb2
import google.protobuf as pb
from caffe import layers as L
from caffe import params as P
import caffe
from copy import deepcopy

class ConvBNReLULego(BaseLego):
    type = 'ConvBNReLU'
    def __init__(self, params):
        # self._required = ['name', 'kernel_size', 'num_output', 'pad', 'stride', 'use_global_stats']
        self._required = ['name', 'num_output', 'use_global_stats']
        self._check_required_params(params)
        self.convParams = deepcopy(params)
        del self.convParams['use_global_stats']
        self.convParams['name'] = 'conv_' + params['name']
        self.batchNormParams = dict(use_global_stats=params['use_global_stats'],
                  name='bn_' + params['name'])
        self.scaleParams = dict(name='scale_' + params['name'])
        self.reluParams = dict(name='relu_' + params['name'])

    def attach(self, netspec, bottom):
        conv = BaseLegoFunction('Convolution', self.convParams).attach(netspec, bottom)
        bn = BaseLegoFunction('BatchNorm', self.batchNormParams).attach(netspec, [conv])
        scale = BaseLegoFunction('Scale', self.scaleParams).attach(netspec, [bn])
        relu = BaseLegoFunction('ReLU', self.reluParams).attach(netspec, [scale])
        return relu

class ConvReLULego(BaseLego):
    def __init__(self, params):
        self._required = ['name', 'kernel_size', 'num_output', 'pad', 'stride']
        self._check_required_params(params)
        self.convParams = deepcopy(params)
        self.convParams['name'] = 'conv' + params['name']
        self.reluParams = dict(name='relu' + params['name'])

    def attach(self, netspec, bottom):
        conv = BaseLegoFunction('Convolution', self.convParams).attach(netspec, bottom)
        relu = BaseLegoFunction('ReLU', self.reluParams).attach(netspec, [conv])
        return relu



class ConvBNLego(BaseLego):
    type = 'ConvBN'
    def __init__(self, params):
        self._required = ['name', 'kernel_size', 'num_output', 'pad', 'stride', 'use_global_stats']
        self._check_required_params(params)
        self.convParams = deepcopy(params)
        del self.convParams['use_global_stats']
        self.convParams['name'] = 'conv_' + params['name']
        self.batchNormParams = dict(use_global_stats=params['use_global_stats'],
                  name='bn_' + params['name'])
        self.scaleParams = dict(name='scale_' + params['name'])


    def attach(self, netspec, bottom):
        conv = BaseLegoFunction('Convolution', self.convParams).attach(netspec, bottom)
        bn = BaseLegoFunction('BatchNorm', self.batchNormParams).attach(netspec, [conv])
        scale = BaseLegoFunction('Scale', self.scaleParams).attach(netspec, [bn])
        return scale

class EltwiseReLULego(BaseLego):
    type = 'EltwiseReLU'
    def __init__(self, params):
        self._required = ["name"]
        self._check_required_params(params)
        self.eltwiseParams = dict(name='eltwise_' + params['name'])
        self.reluParams = dict(name='relu_' + params['name'])

    def attach(self, netspec, bottom):
        eltwise = BaseLegoFunction('Eltwise', self.eltwiseParams).attach(netspec, bottom)
        relu = BaseLegoFunction('ReLU', self.reluParams).attach(netspec, [eltwise])
        return relu


class FireLego(BaseLego):
    type = 'Fire'
    def __init__(self, params):
        self._required = ['name', 'squeeze_num_output', 'use_global_stats']
        self._check_required_params(params)
        self.name = params['name']
        self.squeeze_num_output = params['squeeze_num_output']
        self.use_global_stats = params['use_global_stats']
        self.downsample = True if 'downsample' in params else False
        self.filter_mult = 4
        if 'filter_mult' in params:
            self.filter_mult = params['filter_mult']


    def attach(self, netspec, bottom):
        # Squeeze
        name = self.name + '_' + 'squeeze_1by1'
        sq_params = dict(name=name, num_output=self.squeeze_num_output,
                                 kernel_size=1, pad=0, stride=1,
                                 use_global_stats=self.use_global_stats)
        sq = ConvBNReLULego(sq_params).attach(netspec, bottom)

        stride = 2 if self.downsample  else 1

        # expand
        name = self.name + '_' + 'expand_1by1'
        exp1_params = dict(name=name, num_output=self.squeeze_num_output * self.filter_mult,
                                 kernel_size=1, pad=0, stride=stride,
                                 use_global_stats=self.use_global_stats)
        exp1 = ConvBNReLULego(exp1_params).attach(netspec, [sq])

        name = self.name + '_' + 'expand_3by3'
        exp2_params = dict(name=name, num_output=self.squeeze_num_output * self.filter_mult,
                                 kernel_size=3, pad=1, stride=stride,
                                 use_global_stats=self.use_global_stats)
        exp2 = ConvBNReLULego(exp2_params).attach(netspec, [sq])

        # concat
        name = self.name + '_' + 'concat'
        concat = BaseLegoFunction('Concat', dict(name=name)).attach(netspec, [exp1, exp2])

        return concat


class InceptionV1Lego(BaseLego):
    type = 'InceptionV1'
    def __init__(self, params):
        self._required = ['name', 'num_outputs', 'use_global_stats']
        self._check_required_params(params)
        self.name = params['name']
        self.num_outputs = params['num_outputs']
        self.use_global_stats = params['use_global_stats']

    def attach(self, netspec, bottom):
        # branch1by1
        name = self.name + '_' + 'br1by1'
        params = dict(name=name, num_output=self.num_outputs[0],
                                 kernel_size=1, pad=0, stride=1,
                                 use_global_stats=self.use_global_stats)
        br1by1 = ConvBNReLULego(params).attach(netspec, bottom)

        # branch 3by3
        name = self.name + '_' + 'br3by3_reduce'
        params = dict(name=name, num_output=self.num_outputs[1],
                                 kernel_size=1, pad=0, stride=1,
                                 use_global_stats=self.use_global_stats)
        br3by3_reduce = ConvBNReLULego(params).attach(netspec, bottom)

        name = self.name + '_' + 'br3by3_expand'
        params = dict(name=name, num_output=self.num_outputs[2],
                                 kernel_size=3, pad=1, stride=1,
                                 use_global_stats=self.use_global_stats)
        br3by3_expand = ConvBNReLULego(params).attach(netspec, [br3by3_reduce])

        # branch 5by5
        name = self.name + '_' + 'br5by5_reduce'
        params = dict(name=name, num_output=self.num_outputs[3],
                                 kernel_size=1, pad=0, stride=1,
                                 use_global_stats=self.use_global_stats)
        br5by5_reduce = ConvBNReLULego(params).attach(netspec, bottom)

        name = self.name + '_' + 'br5by5_expand'
        params = dict(name=name, num_output=self.num_outputs[4],
                                 kernel_size=5, pad=2, stride=1,
                                 use_global_stats=self.use_global_stats)
        br5by5_expand = ConvBNReLULego(params).attach(netspec, [br5by5_reduce])

        # branch pool
        name = self.name + '_' + 'pool_reduce'
        params = dict(kernel_size=3, stride=1, pool=P.Pooling.MAX, name=name)
        pool = BaseLegoFunction('Pooling', params).attach(netspec, bottom)

        name = self.name + '_' + 'pool_expand'
        params = dict(name=name, num_output=self.num_outputs[5],
                                 kernel_size=1, pad=1, stride=1,
                                 use_global_stats=self.use_global_stats)
        pool_conv = ConvBNReLULego(params).attach(netspec, [pool])

        # concat
        name = self.name + '_' + 'concat'
        concat = BaseLegoFunction('Concat', dict(name=name)).attach(netspec,
                                [br1by1, br3by3_expand, br5by5_expand, pool_conv ])
        return concat


class InceptionLego(BaseLego):
    def __init__(self, params):
        self._required = ['name', 'num_output', 'use_global_stats', 'downsample']
        self._check_required_params(params)
        self.name = params['name']
        self.num_output = params['num_output']
        self.use_global_stats = params['use_global_stats']
        self.downsample = params['downsample']

    def attach(self, netspec, bottom):
        stride = 2 if self.downsample  else 1

        # branch1by1
        name = self.name + '_' + 'br1by1'
        params = dict(name=name, num_output=self.num_output / 4,
                                 kernel_size=1, pad=0, stride=stride,
                                 use_global_stats=self.use_global_stats)
        br1by1 = ConvBNReLULego(params).attach(netspec, bottom)

        # branch 3by3
        name = self.name + '_' + 'br3by3_reduce'
        params = dict(name=name, num_output=self.num_output * 3 / 16,
                                 kernel_size=1, pad=0, stride=1,
                                 use_global_stats=self.use_global_stats)
        br3by3_reduce = ConvBNReLULego(params).attach(netspec, bottom)

        name = self.name + '_' + 'br3by3_expand'
        params = dict(name=name, num_output=self.num_output / 4,
                                 kernel_size=3, pad=1, stride=stride,
                                 use_global_stats=self.use_global_stats)
        br3by3_expand = ConvBNReLULego(params).attach(netspec, [br3by3_reduce])

        # branch 2*3by3
        name = self.name + '_' + 'br2_3by3_reduce'
        params = dict(name=name, num_output=self.num_output * 3 / 16,
                                 kernel_size=1, pad=0, stride=1,
                                 use_global_stats=self.use_global_stats)
        br2_3by3_reduce = ConvBNReLULego(params).attach(netspec, bottom)

        name = self.name + '_' + 'br2_3by3_expand_1'
        params = dict(name=name, num_output=self.num_output / 4,
                                 kernel_size=3, pad=1, stride=1,
                                 use_global_stats=self.use_global_stats)
        br2_3by3_expand_1 = ConvBNReLULego(params).attach(netspec, [br2_3by3_reduce])
        name = self.name + '_' + 'br2_3by3_expand_2'
        params = dict(name=name, num_output=self.num_output / 4,
                                 kernel_size=3, pad=1, stride=stride,
                                 use_global_stats=self.use_global_stats)
        br2_3by3_expand_2 = ConvBNReLULego(params).attach(netspec, [br2_3by3_expand_1])


        # branch pool
        name = self.name + '_' + 'pool'
        pad = 0 if self.downsample  else 1
        params = dict(kernel_size=3, stride=stride, pool='max', name=name, pad=pad)
        pool = BaseLegoFunction('Pooling', params).attach(netspec, bottom)

        name = self.name + '_' + 'pool_expand'
        params = dict(name=name, num_output=self.num_output / 4,
                                 kernel_size=1, pad=0, stride=1,
                                 use_global_stats=self.use_global_stats)
        pool_conv = ConvBNReLULego(params).attach(netspec, [pool])

        # concat
        name = self.name + '_' + 'concat'
        concat = BaseLegoFunction('Concat', dict(name=name)).attach(netspec, [br1by1, br3by3_expand, br2_3by3_expand_2, pool_conv ])
        # concat = ConcatLego(dict(name=name)).attach(netspec, [br1by1, br3by3_expand, br2_3by3_expand_2 ])
        return concat

class ShortcutLego(BaseLego):
    type = 'Shortcut'
    def __init__(self, params):
        self._required = ['name', 'shortcut', 'main_branch', 'stride', 'num_output', 'use_global_stats']
        self._check_required_params(params)
        self.name = params['name']
        self.shortcut = params['shortcut']
        self.stride = params['stride']
        self.main_branch = params['main_branch']
        self.num_output = params['num_output']
        self.use_global_stats = params['use_global_stats']

    def attach(self, netspec, bottom):


        if self.shortcut == 'identity':
            shortcut = bottom[0]
        elif self.shortcut == 'projection':
            name = self.name + '_proj_shortcut'
            num_output = self.num_output
            shortcut_params = dict(name=name , num_output=num_output,
                                 kernel_size=1, pad=0, stride=self.stride,
                                 use_global_stats=self.use_global_stats)
            shortcut = ConvBNLego(shortcut_params).attach(netspec, bottom)

            # Convolution(kernel_w=3,kernel_h=1,num_output=64,pad_w=1)+BatchNorm+Scale+ReLU+
            # Convolution(kernel_w=1,kernel_h=3,num_output=64,pad_h=1)+BatchNorm+Scale+ReLU
        if self.main_branch == 'inception_trick':
            name = self.name + '_branch_3by1a'
            num_output = self.num_output
            br3by1a_params = dict(name=name, num_output=num_output,
                                 kernel_w=3, kernel_h=1, pad_w=1, stride_w=self.stride, stride_h=1,
                                 use_global_stats=self.use_global_stats)
            br3by1a = ConvBNReLULego(br3by1a_params).attach(netspec, bottom)

            name = self.name + '_branch_1by3a'
            num_output = self.num_output
            br1by3a_params = dict(name=name, num_output=num_output,
                                  kernel_w=1, kernel_h=3, pad_h=1, stride_h=self.stride, stride_w=1,
                                 use_global_stats=self.use_global_stats)
            br1by3a = ConvBNReLULego(br1by3a_params).attach(netspec, [br3by1a])

            name = self.name + '_branch_3by1b'
            br3by1b_params = dict(name=name, num_output=num_output,
                                 kernel_w=3, kernel_h=1, pad_w=1, stride=1,
                                 use_global_stats=self.use_global_stats)
            br3by1a = ConvBNReLULego(br3by1b_params).attach(netspec, [br1by3a])
            name = self.name + '_branch_1by3b'

            br1by3a_params = dict(name=name, num_output=num_output,
                                 kernel_w=1, kernel_h=3, pad_h=1, stride=1,
                                 use_global_stats=self.use_global_stats)
            br2_out = ConvBNReLULego(br1by3a_params).attach(netspec, [br3by1a])

        if self.main_branch == 'inception_trick_bottleneck':
            name = self.name + '_branch2a'
            num_output = self.num_output
            br2a_params = dict(name=name, num_output=num_output / 4,
                                 kernel_size=1, pad=0, stride=self.stride,
                                 use_global_stats=self.use_global_stats)
            br2a = ConvBNReLULego(br2a_params).attach(netspec, bottom)


            name = self.name + '_branch_b_3by1'
            num_output = self.num_output
            br3by1a_params = dict(name=name, num_output=num_output / 4,
                                 kernel_w=3, kernel_h=1, pad_w=1, stride=1,
                                 use_global_stats=self.use_global_stats)
            br3by1a = ConvBNReLULego(br3by1a_params).attach(netspec, [br2a])

            name = self.name + '_branch_b_1by3'
            num_output = self.num_output
            br1by3a_params = dict(name=name, num_output=num_output / 4,
                                  kernel_w=1, kernel_h=3, pad_h=1, stride=1,
                                 use_global_stats=self.use_global_stats)
            br1by3a = ConvBNReLULego(br1by3a_params).attach(netspec, [br3by1a])

            name = self.name + '_branch2c'
            br2c_params = dict(name=name, num_output=num_output,
                                 kernel_size=1, pad=0, stride=1,
                                 use_global_stats=self.use_global_stats)
            br2_out = ConvBNLego(br2c_params).attach(netspec, [br1by3a])


        if self.main_branch == 'bottleneck':
            name = self.name + '_branch2a'
            num_output = self.num_output
            br2a_params = dict(name=name, num_output=num_output / 4,
                                 kernel_size=1, pad=0, stride=self.stride,
                                 use_global_stats=self.use_global_stats)
            br2a = ConvBNReLULego(br2a_params).attach(netspec, bottom)

            name = self.name + '_branch2b'
            br2b_params = dict(name=name, num_output=num_output / 4,
                                 kernel_size=3, pad=1, stride=1,
                                 use_global_stats=self.use_global_stats)
            br2b = ConvBNReLULego(br2b_params).attach(netspec, [br2a])

            name = self.name + '_branch2c'
            br2c_params = dict(name=name, num_output=num_output,
                                 kernel_size=1, pad=0, stride=1,
                                 use_global_stats=self.use_global_stats)
            br2_out = ConvBNLego(br2c_params).attach(netspec, [br2b])

        elif self.main_branch == 'inception':
            name = self.name + '_inception'
            inception_params = dict(name=name, num_output=self.num_output,
                            use_global_stats=self.use_global_stats)
            inception_params['downsample'] = True if self.shortcut == 'projection' else False
            br2_out = InceptionLego(inception_params).attach(netspec, bottom)

        elif self.main_branch == '2inception':
            name = self.name + '_inception_a'
            inception_params_a = dict(name=name, num_output=self.num_output,
                            use_global_stats=self.use_global_stats,
                            downsample=False)
            inception_a = InceptionLego(inception_params_a).attach(netspec, bottom)

            name = self.name + '_inception_b'
            inception_params_b = dict(name=name, num_output=self.num_output,
                            use_global_stats=self.use_global_stats)
            inception_params_b['downsample'] = True if self.shortcut == 'projection' else False
            br2_out = InceptionLego(inception_params_b).attach(netspec, [inception_a])

        elif self.main_branch == 'normal':
            name = self.name + '_branch2a'
            num_output = self.num_output
            br2a_params = dict(name=name, num_output=num_output,
                                 kernel_size=3, pad=1, stride=self.stride,
                                 use_global_stats=self.use_global_stats)
            br2a = ConvBNReLULego(br2a_params).attach(netspec, bottom)

            name = self.name + '_branch2b'
            br2b_params = dict(name=name, num_output=num_output,
                                 kernel_size=3, pad=1, stride=1,
                                 use_global_stats=self.use_global_stats)
            br2_out = ConvBNLego(br2b_params).attach(netspec, [br2a])

        elif self.main_branch == '1by1_normal':
            name = self.name + '_branch2a'
            num_output = self.num_output
            br2a_params = dict(name=name, num_output=num_output,
                                 kernel_size=1, pad=0, stride=self.stride,
                                 use_global_stats=self.use_global_stats)
            br2a = ConvBNReLULego(br2a_params).attach(netspec, bottom)

            name = self.name + '_branch2b'
            br2b_params = dict(name=name, num_output=num_output,
                                 kernel_size=1, pad=0, stride=1,
                                 use_global_stats=self.use_global_stats)
            br2_out = ConvBNLego(br2b_params).attach(netspec, [br2a])

        # Combine the branches using EltwiseRelu lego
        eltrelu_params = dict(name=self.name)
        eltwiseRelu = EltwiseReLULego(eltrelu_params).attach(netspec, [shortcut, br2_out])

        return eltwiseRelu

