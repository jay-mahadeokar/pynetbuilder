"""
Copyright 2016 Yahoo Inc.
Licensed under the terms of the 2 clause BSD license. 
Please see LICENSE file in the project root for terms.
"""

from base import BaseLego

from caffe.proto import caffe_pb2
import google.protobuf as pb
from caffe import layers as L
from caffe import params as P
import caffe

'''
    Generic class to read data layer
    info from config files.
'''
class ConfigDataLego(BaseLego):
    def __init__(self, data_file):
        self.data_file = data_file

    def attach(self, netspec):
        return

class ImageDataLego(BaseLego):
    def __init__(self, params):
        if params['include'] == 'test':
            params['include'] = dict(phase=caffe.TEST)
        elif params['include'] == 'train':
            params['include'] = dict(phase=caffe.TRAIN)
        params['image_data_param'] = dict(source=params['source'] ,
                                          batch_size=params['batch_size'])
        if 'mean_file' in params:
            params['transform_param'] = dict(mean_file=params['mean_file'])
        self._required = ['name', 'source', 'batch_size', 'include']
        super(ImageDataLego, self).__init__(params)

    def _init_default_params(self):
        self._default['ntop'] = 2

    def attach(self, netspec):
        param_packet = self._construct_param_packet()
        data_lego, label_lego = L.ImageData(**param_packet)
        netspec['data'] = data_lego
        netspec['label'] = label_lego
        return data_lego, label_lego

