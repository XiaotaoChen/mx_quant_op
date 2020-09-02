import mxnet as mx 
import numpy as np

from operator_py.quantization_int8 import *

data_shape = (1, 3, 5, 5)
np.random.seed(5)
data = np.random.uniform(size=data_shape).astype('float32')
data = (data - 0.4) * 20

ctx = mx.gpu()

nbits = 4
quant_level = pow(2, nbits) - 1

label_shape = (1,)
label = np.ones(shape=label_shape)
mx_data = mx.nd.array(data)
mx_label = mx.nd.array(label)


data_names = ['data']
label_names = ["softmax_label"]
mx_data_shape = [('data', data_shape)]
mx_label_shape = [("softmax_label", label_shape)]
mx_data_batch = mx.io.DataBatch(data=[mx.nd.array(data)], label=[mx.nd.array(label)])


def get_sym(quant_type="cxx"):
    data_var = mx.sym.Variable(name="data", shape=data_shape)
    weight_var = mx.sym.Variable(name="weight", shape=(2,3,3,3), dtype="float32")

    weight_minmax_var = mx.sym.Variable(name="weight_minmax", init=mx.init.Constant(0), dtype="float32")
    act_minmax_var = mx.sym.Variable(name="act_minmax", init=mx.init.Constant(0), dtype="float32")

    if quant_type == "python":
        weight = mx.sym.Custom(name="conv0_weight", data=weight_var, minmax=weight_minmax_var, 
                             nbits=nbits, quant_mode="minmax", is_weight=True, is_weight_perchannel=True,
                             delay_quant=0, ema_decay=0.99, grad_mode="ste", fix_act_scale=False, bias_correct=True,
                             op_type="Quantization_int8_PY")
    else:
        weight = mx.sym.contrib.Quantization_int8(name="weight", data=weight_var, minmax=weight_minmax_var,
                                                    nbits=nbits, quant_mode="minmax", is_weight=True, is_weight_perchannel=True,
                                                    delay_quant=0, ema_decay=0.99, grad_mode="ste", fix_act_scale=False)

    sym = mx.sym.Convolution(data=data_var, weight=weight, kernel=(3,3), stride=(1,1), no_bias=True, num_filter=2, pad=(1,1))
    act = mx.sym.Activation(data=sym, act_type='relu', name='relu')

    if quant_type == "python":
        act = mx.sym.Custom(name="act", data=act, minmax=act_minmax_var, 
                             nbits=nbits, quant_mode="minmax", is_weight=False, is_weight_perchannel=False,
                             delay_quant=0, ema_decay=0.99, grad_mode="ste", fix_act_scale=False, bias_correct=False,
                             op_type="Quantization_int8_PY")
    else:
        act = mx.sym.contrib.Quantization_int8(name="act", data=act, minmax=act_minmax_var,
                                                    nbits=nbits, quant_mode="minmax", is_weight=False, is_weight_perchannel=False,
                                                    delay_quant=0, ema_decay=0.99, grad_mode="ste", fix_act_scale=False)
    flat = mx.symbol.Flatten(data=act)
    sym = mx.symbol.SoftmaxOutput(data=flat, name='softmax')
    return sym

def get_mod(sym):
    mx_mod = mx.mod.Module(symbol=sym, context=ctx, data_names=data_names, label_names=label_names)
    mx_mod.bind(for_training=True, data_shapes=mx_data_shape, label_shapes=mx_label_shape)
    mx_mod.init_params()
    mx_mod.init_optimizer()
    return mx_mod

def simulate(data, nbits, is_weight, is_weight_perchannel, old_threshold=None):
    data_abs = np.abs(data)
    quant_level = 2**nbits -1
    if is_weight:
        if is_weight_perchannel:
            channel = data.shape[0]
            # thresholds = np.max(data_abs, axis=(1,2,3))
            thresholds = np.array([0.5, 0.7])
            thresholds = thresholds.reshape((channel, 1, 1, 1))
        else:
            # thresholds = np.max(data_abs)
            thresholds = np.array([0.5])
        quant_unit = thresholds / quant_level
        quant_data = np.round(data / quant_unit) * quant_unit
    else:
        # thresholds = np.max(data_abs)
        # thresholds = 0.99 * old_threshold + (1 - 0.99) * thresholds
        thresholds = np.array([0.5])
        data = np.clip(data, -thresholds, thresholds)
        quant_unit = thresholds / quant_level
        quant_data = np.round(data /quant_unit) * quant_unit
    return quant_data
    
def test_nd():
    is_weight = True
    is_weight_perchannel = False

    mx_data = mx.nd.array(data, ctx=ctx)
    if is_weight_perchannel and is_weight:
        mx_minmax = mx.nd.array([0.5, 0.7], ctx=ctx) 
    else:
        mx_minmax = mx.nd.array([0.5], ctx=ctx) 
    # mx_minmax = mx.nd.array([0.5], ctx=ctx) 
    out = mx.nd.contrib.Quantization_int8(name="weight", data=mx_data, minmax=mx_minmax,
                                            nbits=nbits, quant_mode="minmax", is_weight=is_weight, is_weight_perchannel=is_weight_perchannel,
                                            delay_quant=0, ema_decay=0.99, grad_mode="ste", fix_act_scale=False)
    mx.nd.waitall()
    print("mx.out:\n{}".format(out.asnumpy()))
    sim_out = simulate(data, nbits, is_weight, is_weight_perchannel)
    print("np.out:\n{}".format(sim_out))
    print("mx - np:\n{}".format(out.asnumpy() - sim_out))

def test_sym(quant_type="cxx"):
    print("************************ test symbol mode with {} op ************************".format(quant_type))
    network_flag = "test"
    sym = get_sym(quant_type)
    mx_mod = get_mod(sym)
    arg_params, aux_params = mx_mod.get_params()
    print("*************************** inited params ***************************")
    print(arg_params)
    print(aux_params)

    mx_mod.forward(mx_data_batch)
    output = mx_mod.get_outputs()[0].asnumpy()
    mx_mod.backward()
    mx_mod.update()
    mx.nd.waitall()
    
    print("*************************** params updated with 1 iter ***************************")

    arg_params, aux_params = mx_mod.get_params()
    print(arg_params)
    print(aux_params)

    print("output:{}".format(output))

    # mx.model.save_checkpoint(network_flag, 1, sym, arg_params, aux_params)


if __name__ == "__main__":
    test_nd()
    
    # quant_type = "python" # "python" "cxx"
    # test_sym(quant_type)

