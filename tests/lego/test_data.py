from caffe.proto import caffe_pb2
import google.protobuf as pb
from caffe import layers as L
from caffe import params as P
import caffe

import sys
sys.path.append('../netbuilder')

def test_image_data_lego():
    from lego.data import ImageDataLego
    n = caffe.NetSpec()
    params = dict(name='data', source='tmp' , batch_size=100, include='test', mean_file='tmp')
    data, label = ImageDataLego(params).attach(n)
    assert data is not None and label is not None
    # print >> sys.stderr, n.to_proto()

