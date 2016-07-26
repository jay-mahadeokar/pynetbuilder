"""
Copyright 2016 Yahoo Inc.
Licensed under the terms of the 2 clause BSD license. 
Please see LICENSE file in the project root for terms.
"""

"""
    Base class for all legos in netbuilder
    Lego is a wrapper on the caffe Netspec layers.
    Additional features include:
    1. Picking up default params from a config file
    2. Checking required params are passed
    3. Attach function - Every new lego should implement its 
    own attach function
"""
class BaseLego(object):
    def __init__(self, params):
        if self._required is None:
            self._required = []
        self._default = dict()
        self._init_default_params()
        self._check_required_params(params)
        self._required_params = params

    """
        Returns a list of required parameters
        for building the lego
    """
    def get_required_names(self):
        return self._required

    """
        Returns a list of default parameters
        for building the lego
    """
    def get_default_params(self):
        return self._default

    """
        This function loads the default parameters
        corresponding to the core layers from config file.
        Change the config file to according to your params desired.
        You can override the default params using override_default_params()
    """
    def _init_default_params(self):
        '''
        default_params = {}
        execfile("./config/default.params", default_params)

        if self._type_name in default_params:
            self._default = default_params[self._type_name]
        '''
        self._default = Config.get_default_params(self._type_name)


    def _construct_param_packet(self):
        params = self._required_params.copy()
        params.update(self._default)
        return params

    def override_default_param(self, key, val):
        self._default[key] = val

    """
        utility method to make sure the required params
        dependencies are satisfied
        Each subclass' create method should call 
        this method before creating the lego
    """
    def _check_required_params(self, required_params):
        for r in self._required:
            if r not in required_params.keys():
                raise KeyError('Please specify %s since it is a required parameter' % r)
        return True


    """
        Takes in params and makes the caffe layer object.
        @param caffenet: The caffe network specification object on which lego will be attached
        @param bottom: List of the bottom layers needed
        @return: Caffe Layer object modules inside caffe.layers
    """
    def attach(self, netspec, bottom):
        raise NotImplementedError


from caffe import layers as L
from caffe import params as P

class BaseLegoFunction(BaseLego):
    """
    A Functional wrapper on top of netspec Function class
    This is used to attach all the basic layers in caffe
    to a netspec object.
    
    """
    def __init__(self, type_name, params):
        if '_required' not in self.__dict__:
            self._required = []
        self._type_name = type_name
        super(BaseLegoFunction, self).__init__(params)

    """
        Takes in params and makes the caffe layer object.
        @param caffenet: The caffe network specification object on which lego will be attached
        @param bottom: List of the bottom layers needed
        @return: Caffe Layer object modules inside caffe.layers
    """
    def attach(self, netspec, bottom):

        param_packet = self._construct_param_packet()
        layername = getattr(L, self._type_name)
        layer = layername(*bottom, ** param_packet)
        netspec[param_packet['name']] = layer
        return layer


'''
    This class provides a way to read and modify
    the default parameters of the core legos
'''
class Config(object):
    _default_params = {}

    @classmethod
    def get_default_params(self, layer):
        if len(Config._default_params) == 0:
            execfile("./config/default.params", Config._default_params)

        if layer in Config._default_params:
            return Config._default_params[layer]
        else:
            return {}

    @classmethod
    def set_default_params(self, layer, param, val):
        if len(Config._default_params) == 0:
            execfile("./config/default.params", Config._default_params)
        assert layer in Config._default_params
        assert param in Config._default_params[layer]
        Config._default_params[layer][param] = val

