# -*- coding: utf-8 -*- 
"""
File Name: calculate_flops.py
Author: liangdepeng
mail: liangdepeng@gmail.com
"""

import mxnet as mx
import argparse
import numpy as np
import json
import re

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-ds', '--data_shapes', type=str, nargs='+',
                        help='data_shapes, format: arg_name,s1,s2,...,sn, example: data,1,3,224,224')
    parser.add_argument('-ls', '--label_shapes', type=str, nargs='+',
                        help='label_shapes, format: arg_name,s1,s2,...,sn, example: label,1,1,224,224')
    parser.add_argument('-s', '--symbol_path', type=str, default='./caffenet-symbol.json', help='')
    return parser.parse_args()

def product(tu):
    """Calculates the product of a tuple"""
    prod = 1
    for x in tu:
        prod = prod * x
    return prod

def get_internal_label_info(internal_sym, label_shapes):
    if label_shapes:
        internal_label_shapes = filter(lambda shape: shape[0] in internal_sym.list_arguments(), label_shapes)
        if internal_label_shapes:
            internal_label_names = [shape[0] for shape in internal_label_shapes]
            return internal_label_names, internal_label_shapes 
    return None, None

if __name__ == '__main__':
    args = parse_args()
    sym = mx.sym.load(args.symbol_path)
    
    data_shapes = list()
    data_names = list()
    if args.data_shapes is not None and len(args.data_shapes) > 0:
        for shape in args.data_shapes:
            items = shape.replace('\'', '').replace('"', '').split(',')
            data_shapes.append((items[0], tuple([int(s) for s in items[1:]])))
            data_names.append(items[0])
            
    label_shapes = None
    label_names = list()
    if args.label_shapes is not None and len(args.label_shapes) > 0:
        label_shapes = list()
        for shape in args.label_shapes:
            items = shape.replace('\'', '').replace('"', '').split(',')
            label_shapes.append((items[0], tuple([int(s) for s in items[1:]])))
            label_names.append(items[0])

    devs = [mx.cpu()]

    if len(label_names) == 0:
        label_names = None
    model = mx.mod.Module(context=devs, symbol=sym, data_names=data_names,label_names=None)
    model.bind(data_shapes=data_shapes, label_shapes=label_shapes, for_training=False)

    arg_params = model._exec_group.execs[0].arg_dict

    conf = json.loads(sym.tojson())
    nodes = conf["nodes"]

    total_flops=0.

    for node in nodes:
        op = node["op"]
        layer_name = node["name"]
        attrs = None
        if "param" in node:
            attrs = node["param"]
        elif "attrs" in node:
            attrs = node["attrs"]
        else:
            attrs = {}

        if op == 'Convolution':
            internal_sym = sym.get_internals()[layer_name + '_output']
            internal_label_names, internal_label_shapes = get_internal_label_info(internal_sym, label_shapes)

            shape_dict = {}
            for k,v in data_shapes:
                shape_dict[k] = v
            if internal_label_shapes != None:
                for k,v in internal_label_shapes:
                    shape_dict[k] = v

            _, out_shapes, _ = internal_sym.infer_shape(**shape_dict)
            out_shape = out_shapes[0]

            # support conv1d NCW and conv2d NCHW layout
            out_shape_produt = out_shape[2] if len(out_shape) == 3 else out_shape[2] * out_shape[3]
            total_flops += out_shape_produt * product(arg_params[layer_name + '_weight'].shape) * data_shapes[0][1][0]

            if layer_name + "_bias" in arg_params:
                total_flops += product(out_shape)

            del shape_dict

        
        if op == 'Deconvolution':
            input_layer_name = nodes[node["inputs"][0][0]]["name"]

            internal_sym = sym.get_internals()[input_layer_name + '_output']
            internal_label_names, internal_label_shapes = get_internal_label_info(internal_sym, label_shapes)

            shape_dict = {}
            for k,v in data_shapes:
                shape_dict[k] = v
            if internal_label_shapes != None:
                for k,v in internal_label_shapes:
                    shape_dict[k] = v

            _, out_shapes, _ = internal_sym.infer_shape(**shape_dict)
            input_shape = out_shapes[0]
        
            total_flops += input_shape[2] * input_shape[3] * product(arg_params[layer_name + '_weight'].shape) * data_shapes[0][1][0]

            if layer_name + "_bias" in arg_params:
                internal_sym = sym.get_internals()[layer_name + '_output']
                internal_label_names, internal_label_shapes = get_internal_label_info(internal_sym, internal_label_shapes)

                tmp_model = mx.mod.Module(
                    context=devs, symbol=internal_sym, data_names=data_names, label_names=internal_label_names)
                tmp_model.bind(data_shapes=data_shapes, label_shapes=internal_label_shapes, for_training=False)
                # tmp_model.init_params()
                out_shape = tmp_model.get_outputs()[0].shape

                total_flops += product(out_shape)

            del shape_dict

        if op == 'FullyConnected':
            total_flops += product(arg_params[layer_name + '_weight'].shape) * data_shapes[0][1][0]

            if layer_name + '_bias' in arg_params:
                num_hidden = int(attrs['num_hidden'])
                total_flops += num_hidden * data_shapes[0][1][0]

        if op == 'Pooling':
            if "global_pool" in attrs and attrs['global_pool'] == 'True':
                input_layer_name = nodes[node["inputs"][0][0]]["name"]

                internal_sym = sym.get_internals()[input_layer_name + '_output']
                internal_label_names, internal_label_shapes = get_internal_label_info(internal_sym, label_shapes)

                shape_dict = {}
                for k,v in data_shapes:
                    shape_dict[k] = v
                if internal_label_shapes != None:
                    for k,v in internal_label_shapes:
                        shape_dict[k] = v

                _, out_shapes, _ = internal_sym.infer_shape(**shape_dict)
                input_shape = out_shapes[0]
            
                total_flops += product(input_shape)
            else:
                internal_sym = sym.get_internals()[layer_name + '_output']
                internal_label_names, internal_label_shapes = get_internal_label_info(internal_sym, label_shapes)
                
                shape_dict = {}
                for k,v in data_shapes:
                    shape_dict[k] = v
                if internal_label_shapes != None:
                    for k,v in internal_label_shapes:
                        shape_dict[k] = v

                _, out_shapes, _ = internal_sym.infer_shape(**shape_dict)
                out_shape = out_shapes[0]

                n = '\d+'
                kernel = [int(i) for i in re.findall(n, attrs['kernel'])]

                total_flops += product(out_shape) * product(kernel)

            del shape_dict

        if op == 'Activation':
            if attrs['act_type'] == 'relu':
                internal_sym = sym.get_internals()[layer_name + '_output']
                internal_label_names, internal_label_shapes = get_internal_label_info(internal_sym, label_shapes)
                
                shape_dict = {}
                for k,v in data_shapes:
                    shape_dict[k] = v
                if internal_label_shapes != None:
                    for k,v in internal_label_shapes:
                        shape_dict[k] = v

                _, out_shapes, _ = internal_sym.infer_shape(**shape_dict)
                out_shape = out_shapes[0]

                total_flops += product(out_shape)

                del shape_dict

    
    model_size = 0.0
    for k,v in arg_params.items():
        if k not in data_names and k not in label_names:
            print(k)
            print(v.shape)
            model_size += product(v.shape) * np.dtype(v.dtype()).itemsize

    print('flops: ', str(total_flops / 1000000), ' MFLOPS')
    print('model size: ', str(model_size / 1024 / 1024), ' MB')
    
    