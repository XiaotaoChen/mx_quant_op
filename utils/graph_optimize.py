# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import json
import mxnet as mx
import numpy as np

FLOAT32_DTYPE = 0
QUANT_TYPES = ("Quantization_int8", "DoReFa", "PACT", "GDRQ", "FQN")
DEBUG = False

def get_constant(value):
    init_str = '[\"constant\", {\"value\": ' + str(value) + '}]'
    return init_str

def create_quant_node(var, setting, init_dict=None, fake_quant=False):
    if fake_quant:
        return var
    quantize_op_name = setting['quantize_op_name']
    attrs = setting['attrs']
    assert quantize_op_name in ("Quantization_int8", "DoReFa", "PACT", "GDRQ", "FQN")

    quant_node_name = var.name + "_" + quantize_op_name

    if quantize_op_name == "Quantization_int8":
        if init_dict is not None and var.name in init_dict.keys():
            init_value = init_dict[var.name]
        else:
            init_value = setting['init_value'] or 0
        minmax_var = mx.sym.var(name = var.name + "_minmax", init=mx.init.Constant(init_value))
        quanted_node = mx.sym.contrib.Quantization_int8(name=quant_node_name, data=var, minmax=minmax_var, **attrs)
    elif quantize_op_name == "DoReFa":
        quanted_node = mx.sym.contrib.DoReFa(name=quant_node_name, data=var, **attrs)
    elif quantize_op_name == "PACT":
        if init_dict is not None and var.name in init_dict.keys():
            init_value = init_dict[var.name]
        else:
            init_value = setting['init_value'] or 8.0
        gamma_var = mx.sym.var(name = var.name + "_pact_gamma", init=get_constant(init_value))
        quanted_node = mx.sym.contrib.PACT(name=quant_node_name, data=var, gamma=gamma_var, **attrs)
    elif quantize_op_name == "GDRQ":
        if init_dict is not None and var.name in init_dict.keys():
            init_value = init_dict[var.name]
        else:
            init_value = setting['init_value'] or 1.0
        alpha_var = mx.sym.Variable(name=var.name + "_alpha", init=mx.init.Constant(init_value), dtype="float32")
        quanted_node = mx.sym.contrib.GDRQ(name=quant_node_name, data=var, alpha=alpha_var, **attrs)
    elif quantize_op_name == "FQN":
        quanted_node = mx.sym.contrib.FQN(name=quant_node_name, data=var, **attrs)
    return quanted_node

def attach_quantize_node(symbol, out_shape_dict, weight_setting, act_setting, 
                         quantized_op=None, 
                         skip_quantize_counts=None, quantize_counts=None, 
                         ignored_node=None, init_dict=None, fixbn=False, fake_quant=False, ignore_scalar=True):
    """
    Adapted from https://github.com/dmlc/tvm/blob/master/python/tvm/relay/frontend/mxnet.py
    Instead of translating nnvm graph into TVM relay graph, we adapt the script to translate
    it back to mxnet graph.
    """
    assert symbol is not None
    assert isinstance(weight_setting, dict) and isinstance(act_setting, dict) \
           and isinstance(skip_quantize_counts, dict) and isinstance(quantize_counts, dict)
    if init_dict is not None:
        for k, v in init_dict.items():
            print("{}: {}".format(k, v))
            print("{}: {}".format(k, v))

    quantized_op = quantized_op or ("Convolution", "FullyConnected", "Deconvolution",
                                    "Concat", "concat", "Pooling", "add_n", "elemwise_add")

    jgraph = json.loads(symbol.tojson())
    jnodes = jgraph["nodes"]
    node_map = {}
    node_op_map = {}
    quantized_node_map = {}
    
    ignored_node = ignored_node or []

    print("skip quantize_count:{}".format(skip_quantize_counts))
    print("quantize_count:{}".format(quantize_counts))
    print("weight setting:{}".format(weight_setting['attrs']))
    print("act setting:{}".format(act_setting['attrs']))

    print("ignored node:{}".format(ignored_node))


    visited_op_counts = {"Convolution": 0, "FullyConnected": 0, "Deconvolution": 0, 
                          "Concat": 0, "Pooling": 0, "add_n": 0, "elemwise_add": 0}

    for nid, node in enumerate(jnodes):
        # edges are [which_node, which_output, type(? not sure)]
        # mx.symbol has an attribute of __getitem__. sym[1] gives the second output
        children = [node_map[e[0]][e[1]] for e in node["inputs"]]
        attrs = node.get("attrs", {})
        node_name = node["name"]
        op_name = node["op"]
        if op_name == "null":
            attrs = dict({k:v for k, v in attrs.items() if k.startswith("__")})
            assert node_name in out_shape_dict.keys(), "{} Variable is not in shape_dict".format(node_name)
            if "__shape__" not in attrs.keys():
                attrs["__shape__"] = out_shape_dict[node_name]
                attrs["__dtype__"] = FLOAT32_DTYPE
            elif attrs["__shape__"] != out_shape_dict[node_name]:
                if DEBUG:
                    print("{} source shape {} is not equal to {}, rewrite it!!".format(node_name, attrs["__shape__"], out_shape_dict[node_name]))
                attrs["__shape__"] = out_shape_dict[node_name]
                attrs["__dtype__"] = FLOAT32_DTYPE

            node_map[nid] = mx.sym.var(node_name, **attrs)

            node_op_map[nid] = ["Variable"]
        elif op_name in quantized_op:
            if op_name not in visited_op_counts.keys():
                visited_op_counts[op_name] = 1
            else:
                visited_op_counts[op_name] += 1
            
            included_scalar = False
            for child in children:
                child_name = child.name
                if child_name in out_shape_dict.keys():
                    if out_shape_dict[child_name] == (1,):
                        included_scalar = True
                        break
                else:
                    assert child_name + "_output" in out_shape_dict.keys(), "{} or {}_output not in out_shape_dict".format(child_name, child_name)
                    if out_shape_dict[child.name + "_output"] == (1,):
                        included_scalar = True
                        break
            if ignore_scalar and included_scalar:
                print("to ignore {} due to its'child includes scalar var".format(node_name))
                quanted_children = children
            elif node_name in ignored_node:
                print("ignored node {}".format(node_name))
                quanted_children = children
            # the idx of quantized_op to skip
            elif skip_quantize_counts is not None and op_name in skip_quantize_counts.keys() and \
                visited_op_counts[op_name] <= skip_quantize_counts[op_name]:
                print("skip idx:{} {} on {}".format(visited_op_counts[op_name], op_name, node_name))
                quanted_children = children
            elif quantize_counts is not None and op_name in quantize_counts.keys() and \
                visited_op_counts[op_name] > quantize_counts[op_name]:
                print("reach quantize threshold, skip idx:{} {} on {}".format(visited_op_counts[op_name], op_name, node_name))
                quanted_children = children
            
            elif op_name in ["Convolution", "FullyConnected", "Deconvolution"]:
                if len(children) == 2:
                    datavar, weightvar = children
                    biasvar = None
                else:
                    datavar, weightvar, biasvar = children
                data_name, weight_name = datavar.name, weightvar.name
                if data_name in quantized_node_map.keys():
                    print("{} has attached quantized node".format(data_name))
                    data_quanted = quantized_node_map[data_name]
                else:
                    data_quanted = create_quant_node(datavar, act_setting, init_dict=init_dict, fake_quant=fake_quant)
                    quantized_node_map[data_name] = data_quanted

                if weight_name in quantized_node_map.keys():
                    print("{} has attached quantized node".format(weight_name))
                    weight_quanted = quantized_node_map[weight_name]
                else:
                    weight_quanted = create_quant_node(weightvar, weight_setting, init_dict=None, fake_quant=fake_quant)
                    quantized_node_map[weight_name] = weight_quanted
                print("attach quantize node for {} inputs:{}, {}".format(op_name, data_name, weight_name))
                quanted_children = [data_quanted, weight_quanted, biasvar]
            elif op_name in ["Concat", "concat", "Pooling", "add_n", "elemwise_add", "broadcast_mul", "mean", "max",\
                             "relu", "Activation", "broadcast_add", "sigmoid", "Dropout"]:
                quant_names = [var.name for var in children]
                print("attach quantize node for {} inputs:{}".format(op_name, quant_names))
                quanted_children = [None] * len(children)
                for i, var in enumerate(children):
                    if var.name in quantized_node_map.keys():
                        print("{} has attached quantized node".format(var.name))
                        quanted_children[i] = quantized_node_map[var.name]
                    else:
                        quanted_var = create_quant_node(var, act_setting, init_dict=init_dict, fake_quant=fake_quant)
                        quantized_node_map[var.name] = quanted_var
                        quanted_children[i] = quantized_node_map[var.name]

            else:
                print("Warning {} don't support quantization training currently.".format(op_name))
                quanted_children = children
            operator = eval("mx.sym." + op_name)
            res = operator(*quanted_children, **attrs, name=node_name)
            node_map[nid] = res
            node_op_map[nid] = [op_name]
        else:
            # print("Warning {} don't support quantization training currently.".format(op_name))
            if op_name.startswith("_contrib_"):
                op_name = op_name.replace("_contrib_", "")
                operator = eval("mx.sym.contrib." + op_name)
            elif op_name.startswith("_np_"):
                op_name = op_name.replace("_np_", "")
                operator = eval("mx.sym." + op_name)
            elif op_name.startswith("_"):
                operator = eval("mx.sym._internal." + op_name)
            else:
                operator = eval("mx.sym." + op_name)
            if fixbn and "BatchNorm" in op_name:
                attrs["use_global_stats"] = True
                # if "SyncBatchNorm" in op_name:
                #     del attrs['key']
                #     del attrs['ndev']
                #     operator = mx.sym.BatchNorm
            res = operator(*children, **attrs, name=node_name)
            node_map[nid] = res
            node_op_map[nid] = [op_name]

    outputs = [node_map[e[0]][e[1]] for e in jgraph["heads"]]
    outputs = outputs[0] if len(outputs) == 1 else mx.sym.Group(outputs)
    return outputs

def mergebn_for_deploy(symbol, args, auxs):
    """
    Adapted from https://github.com/dmlc/tvm/blob/master/python/tvm/relay/frontend/mxnet.py
    Instead of translating nnvm graph into TVM relay graph, we adapt the script to translate
    it back to mxnet graph.
    """
    assert symbol is not None
    jgraph = json.loads(symbol.tojson())
    jnodes = jgraph["nodes"]
    node_map = {}
    node_op_map = {}

    # added by cxt
    nid2node = {}

    for nid, node in enumerate(jnodes):
        # edges are [which_node, which_output, type(? not sure)]
        # mx.symbol has an attribute of __getitem__. sym[1] gives the second output
        children = [node_map[e[0]][e[1]] for e in node["inputs"]]
        attrs = node.get("attrs", {})
        node_name = node["name"]
        op_name = node["op"]

        nid2node[nid] = node

        if op_name == "null":
            attrs = dict({k:v for k, v in attrs.items() if k.startswith("__")})
            node_map[nid] = mx.sym.var(node_name, **attrs)
            node_op_map[nid] = ["Variable"]

        elif op_name == "BatchNorm" and  node_op_map[node["inputs"][0][0]][node["inputs"][0][1]] == "Convolution":
            e = node["inputs"][0]

            conv_nid = e[0]
            conv_node = nid2node[conv_nid]
            conv_attrs = conv_node.get("attrs", {}).copy()
            if  "no_bias" in conv_attrs.keys() and conv_attrs["no_bias"] == "True":
                conv_attrs["no_bias"] = "False"

            if args is not None and auxs is not None:
                _, gamma, beta, mmean, mvar = children
                gamma_name, beta_name, mmean_name, mvar_name = gamma.name, beta.name, mmean.name, mvar.name
                assert "gamma" in gamma_name
                assert "beta" in beta_name
                assert "moving_mean" in mmean_name
                assert "moving_var" in mvar_name
                eps = float(attrs["eps"])
                assert mmean_name in auxs.keys() and mvar_name in auxs.keys(), "{}/{} can't found mean/var name in auxs".format(mmean_name, mvar_name)
                # get conv weight
                conv_w_name = conv_node['name'] + '_weight'
                assert conv_w_name in args.keys(), "{} not in args".format(conv_w_name)
                # modify beta before gamma since gamma is not depend on beta
                args[beta_name] -= args[gamma_name] * auxs[mmean_name] / mx.nd.sqrt(eps + auxs[mvar_name])
                args[gamma_name] /= mx.nd.sqrt(eps + auxs[mvar_name])

                assert args[conv_w_name].shape[0] == args[beta_name].shape[0], "weight shape \
                vs bn_beta shape:{} vs {}".format(args[conv_w_name].shape, args[beta_name].shape)
        
                print("Merging {} and {}".format(conv_node['name'], node_name))
                # update conv bias
                conv_bias_name = conv_node['name'] + '_bias'
                tmp_attrs = conv_node.get("attrs", {}).copy()
                if "no_bias" not in tmp_attrs.keys() or tmp_attrs["no_bias"] == "False":
                    assert conv_bias_name in args.keys()
                    args[conv_bias_name] = args[conv_bias_name] * args[gamma_name] + args[beta_name]
                elif tmp_attrs["no_bias"] == "True":
                    args[conv_bias_name] = args[beta_name].copy()
                # expand for broadcasting for conv weight
                args[gamma_name] = args[gamma_name].expand_dims(axis=-1).expand_dims(axis=-1).expand_dims(axis=-1)
                # multiple gamma to weight
                args[conv_w_name][:] = args[conv_w_name] * args[gamma_name]
                # delete mean,var in auxs, gamma in args
                del auxs[mmean_name], auxs[mvar_name], args[gamma_name]
                del args[beta_name]

            # create new conv with bias
            conv_children = [node_map[e[0]][e[1]] for e in conv_node["inputs"]]
            res = mx.sym.Convolution(*conv_children, **conv_attrs, name=conv_node['name'])
            node_map[nid] = res
            node_op_map[nid] = ["Convolution"]

        else:
            if op_name.startswith("_contrib_"):
                op_name = op_name.replace("_contrib_", "")
                operator = eval("mx.sym.contrib." + op_name)
            elif op_name.startswith("_"):
                operator = eval("mx.sym._internal." + op_name)
            else:
                operator = eval("mx.sym." + op_name)
            res = operator(*children, **attrs, name=node_name)
            node_map[nid] = res
            node_op_map[nid] = [op_name]

    outputs = [node_map[e[0]][e[1]] for e in jgraph["heads"]]
    outputs = outputs[0] if len(outputs) == 1 else mx.sym.Group(outputs)
    return outputs, args, auxs


if __name__ == "__main__":
    sym = mx.sym.load("sources/resnet18.json")
    worker_data_shape = {"data":(1, 3, 224, 224)}
    quantized_op = ("Convolution", "FullyConnected", "Deconvolution",)
    skip_quantize_counts = {} # {"Convolution": 0, "FullyConnected":0}
    quantize_counts = {} # {"Convolution": 1000, "FullyConnected":1000}

    weight_setting = {
        "quantize_op_name": "Quantization_int8",
        "init_value": None,
        "attrs": {
            "nbits": 7,
            "delay_quant": 0,
            "ema_decay": 0.99,
            "grad_mode": "ste",
            "is_weight": True,
            "is_weight_perchannel": True,
            "fix_act_scale": False,
            "quant_mode": "minmax",
        },
    }
    act_setting = {
        "quantize_op_name": "Quantization_int8",
        "init_value": None,
        "attrs": {
            "nbits": 7,
            "delay_quant": 0,
            "ema_decay": 0.99,
            "grad_mode": "ste",
            "is_weight": False,
            "is_weight_perchannel": False,
            "fix_act_scale": False,
            "quant_mode": "minmax",
        },
    }


    _, out_shape, _ = sym.get_internals().infer_shape(**worker_data_shape)
    out_shape_dictoinary = dict(zip(sym.get_internals().list_outputs(), out_shape))

    sym = attach_quantize_node(sym, out_shape_dictoinary, weight_setting, act_setting, 
                               quantized_op=quantized_op, skip_quantize_counts=skip_quantize_counts,
                               quantize_counts=quantize_counts,)
    sym.save("quantized_sym.json")
