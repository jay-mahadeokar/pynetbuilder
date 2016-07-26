from caffe.proto import caffe_pb2
import google.protobuf as pb
from caffe import layers as L
from caffe import params as P
import caffe

import sys
sys.path.append('./netbuilder')

def test_prediction_lego():
    from lego.ssd import PredictionLego
    n = caffe.NetSpec()
    n.data, n.label = L.ImageData(image_data_param=dict(source='tmp' , batch_size=100),
                                   ntop=2, transform_param=dict(mean_file='tmp'))
    params = dict(name='1', num_output=16, stride=1)
    lego = PredictionLego(params)
    lego.attach(n, [n.data])

