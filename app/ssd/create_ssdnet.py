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
parser = ArgumentParser(description=""" This script generates ssd vggnet train_val.prototxt files""")
parser.add_argument('-t', '--type', help="""Resnet or VGGnet""")
parser.add_argument('-o', '--output_folder', help="""Train and Test prototxt will be generated as train.prototxt and test.prototxt""")
parser.add_argument('-n', '--num_output_stage1', help="""Number of filters in stage 1 of resnet""", type=int, default=128)
parser.add_argument('--mbox_source_layers', nargs='+', help="""Names of layers where detection heads will be attached""")
parser.add_argument('-b', '--blocks', type=int, nargs='+', help="""Number of Blocks in the 4 resnet stages""", default=[3, 4, 6, 3])
parser.add_argument('--extra_blocks', type=int, nargs='+', help="""Number of extra Blocks to be attached to Detection network""", default=[3, 3])
parser.add_argument('--extra_num_outputs', type=int, nargs='+', help="""Number of outputs of extra blocks Detection network""", default=[1024, 1024])
parser.add_argument('--extra_layer_attach', help="""Name of layer where extra Blocks will be attached""")
parser.add_argument('--fc_layers', dest='fc_layers', action='store_true')
parser.add_argument('--no-fc_layers', dest='fc_layers', action='store_false')
parser.set_defaults(fc_layers=False)
parser.add_argument('-m', '--main_branch', help="""normal, bottleneck""", required=True)
parser.add_argument('-c', '--num_classes', help="""Number of classes in detection dataset""", type=int, default=21)


from tools.complexity import get_complexity
from nets.ssdnet import get_vgg_ssdnet, get_resnet_ssdnet

if __name__ == '__main__':

    args = parser.parse_args()

    # source, main_branch, num_output_stage1, fc_layers, blocks
    res_params = dict(main_branch=args.main_branch,
                  num_output_stage1=args.num_output_stage1,
                  blocks=args.blocks,
                  fc_layers=args.fc_layers,
                  extra_blocks=args.extra_blocks,
                  extra_num_outputs=args.extra_num_outputs,
                  mbox_source_layers=args.mbox_source_layers,
                  extra_layer_attach=args.extra_layer_attach,
                  num_classes=args.num_classes)


    print args.type
    if args.type == 'VGG':
        netspec = get_vgg_ssdnet(is_train=True)
    else:
        res_params['is_train'] = True
        netspec = get_resnet_ssdnet(res_params)

    from tools.complexity import get_complexity
    params, flops = get_complexity(netspec=netspec)
    print 'Number of params: ', (1.0 * params) / 1000000.0, ' Million'
    print 'Number of flops: ', (1.0 * flops) / 1000000.0, ' Million'

    fp = open(args.output_folder + '/train.prototxt', 'w')
    print >> fp, netspec.to_proto()
    fp.close()

    if args.type == 'VGG':
        netspec = get_vgg_ssdnet(is_train=False)
    else:
        res_params['is_train'] = False
        netspec = get_resnet_ssdnet(res_params)

    fp = open(args.output_folder + '/test.prototxt', 'w')
    print >> fp, netspec.to_proto()
    fp.close()
