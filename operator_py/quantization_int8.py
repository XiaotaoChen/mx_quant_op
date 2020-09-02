import mxnet as mx
import numpy as np

class Quantization_int8(mx.operator.CustomOp):
    def __init__(self, 
                 nbits, 
                 quant_mode, 
                 is_weight, 
                 is_weight_perchannel, 
                 delay_quant, 
                 ema_decay, 
                 grad_mode, 
                 fix_act_scale,
                 bias_correct):
        self.nbits = nbits
        self.quant_mode = quant_mode
        self.is_weight = is_weight
        self.is_weight_perchannel = is_weight_perchannel
        self.delay_quant = delay_quant
        self.ema_decay = ema_decay
        self.grad_mode = grad_mode
        self.fix_act_scale = fix_act_scale
        self.bias_correct = bias_correct
        self.QUANT_LEVEL = 2**self.nbits -1

        self.inited = False
        
    def forward(self, is_train, req, in_data, out_data, aux):
        if is_train and self.delay_quant > 0:
            self.assign(out_data[0], req[0], in_data[0])
            self.delay_quant -= 1
            return

        if self.is_weight:
            if self.is_weight_perchannel:
                # save weight maxs
                if is_train > 0:
                    data = mx.nd.abs(in_data[0])
                    reduce_axis = tuple([i for i in range(len(data.shape))])
                    maxs = mx.nd.max(data, axis=reduce_axis[1:])
                    # to avoid the max vaule is zero.
                    maxs += 1e-6
                    aux[0][:] = maxs
                target_shape = (in_data[0].shape[0],) + (1,) * len(in_data[0].shape[1:])
                quant_unit = aux[0] / self.QUANT_LEVEL
                quant_unit = quant_unit.reshape(target_shape).broadcast_like(in_data[0])
            else:
                if is_train > 0:
                    data = mx.nd.abs(in_data[0])
                    maxs = mx.nd.max(data)
                    aux[0][:] = maxs
                quant_unit = aux[0] / self.QUANT_LEVEL
            
            quanted_data = mx.nd.round(in_data[0] / quant_unit) * quant_unit

            if self.bias_correct:
                per_channel = True
                if per_channel:
                    data_shape = in_data[0].shape
                    reduce_axis = tuple([i for i in range(len(data_shape))])
                    target_shape = (data_shape[0],) + (1,) * len(data_shape[1:])

                    data_mean = mx.nd.mean(in_data[0], axis=reduce_axis[1:]).reshape(target_shape)
                    # data_norm = mx.nd.norm(in_data[0] - data_mean, axis=reduce_axis[1:]).reshape(target_shape)
                    quanted_mean = mx.nd.mean(quanted_data, axis=reduce_axis[1:]).reshape(target_shape)
                    # quanted_norm = mx.nd.norm(quanted_data - quanted_mean, axis=reduce_axis[1:]).reshape(target_shape)
                else:
                    data_mean = mx.nd.mean(in_data[0])
                    # data_norm = mx.nd.norm(in_data[0]-data_mean)
                    quanted_mean = mx.nd.mean(quanted_data)
                    # quanted_norm = mx.nd.norm(quanted_data-quanted_mean)
                # quanted_data = (data_norm / quanted_norm) * (quanted_data + (data_mean - quanted_mean))
                quanted_data = quanted_data + (data_mean - quanted_mean)

            self.assign(out_data[0], req[0], quanted_data)
        else:
            if is_train and self.fix_act_scale is False:
                data = mx.nd.abs(in_data[0])
                maxs = mx.nd.max(data)
                # udpate activation maxs
                # check the value of minmax in aux is  equal to 0, so this should be initialized. otherwise, the value of
                # minmax load from checkpoint, this don't need init.
                if (aux[0].asnumpy()[0] < 1e-6):
                    aux[0][:] = maxs
                else:
                    aux[0][:] = aux[0] * self.ema_decay + maxs * (1 - self.ema_decay)
            quant_unit = aux[0] / self.QUANT_LEVEL

            # print("act aux:{}, max:{}".format(aux, maxs))

            # out_data[0][:] = mx.nd.clip(in_data[0], 
            #                             - aux[0].asnumpy()[0], 
            #                             aux[0].asnumpy()[0])
            # out_data[0][:] = mx.nd.round(out_data[0] / quant_unit) * quant_unit
            quanted_data =  mx.nd.clip(in_data[0], 
                                        - aux[0].asnumpy()[0], 
                                        aux[0].asnumpy()[0])
            quanted_data = mx.nd.round(quanted_data / quant_unit) * quant_unit

            if self.bias_correct:
                per_channel = True
                if per_channel:
                    data_shape = in_data[0].shape
                    reduce_axis = tuple([i for i in range(len(data_shape))])
                    target_shape = (data_shape[0],) + (1,) * len(data_shape[1:])

                    data_mean = mx.nd.mean(in_data[0], axis=reduce_axis[1:]).reshape(target_shape)
                    # data_norm = mx.nd.norm(in_data[0] - data_mean, axis=reduce_axis[1:]).reshape(target_shape)                    
                    quanted_mean = mx.nd.mean(quanted_data, axis=reduce_axis[1:]).reshape(target_shape)
                    # quanted_norm = mx.nd.norm(quanted_data - quanted_mean, axis=reduce_axis[1:]).reshape(target_shape)
                else:
                    data_mean = mx.nd.mean(in_data[0])
                    # data_norm = mx.nd.norm(in_data[0]-data_mean)
                    quanted_mean = mx.nd.mean(quanted_data)
                    # quanted_norm = mx.nd.norm(quanted_data-quanted_mean)
                
                # quanted_data = (data_norm / quanted_norm) * (quanted_data + (data_mean - quanted_mean))
                quanted_data = quanted_data + (data_mean - quanted_mean)
            self.assign(out_data[0], req[0], quanted_data)




    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        if self.is_weight is False and self.grad_mode == "clip":
            self.assign(in_grad[0], req[0], out_grad[0] * (mx.nd.abs(in_data[0]) < aux[0]))
        else:
            self.assign(in_grad[0], req[0], out_grad[0])

@mx.operator.register("Quantization_int8_PY")
class QuantizationInt8Prop(mx.operator.CustomOpProp):
    def __init__(self, 
                 nbits, 
                 quant_mode, 
                 is_weight, 
                 is_weight_perchannel=False, 
                 delay_quant=0, 
                 ema_decay=0.99, 
                 grad_mode="ste", 
                 fix_act_scale=False,
                 bias_correct=False):
        self.nbits = int(nbits)
        self.quant_mode = str(quant_mode)
        self.is_weight = eval(is_weight)
        self.is_weight_perchannel = eval(is_weight_perchannel)
        self.delay_quant = int(delay_quant)
        self.ema_decay = float(ema_decay)
        self.grad_mode = str(grad_mode)
        self.fix_act_scale = eval(fix_act_scale)
        self.bias_correct = eval(bias_correct)
        
        super(QuantizationInt8Prop, self).__init__(True)
    def list_arguments(self):
        return ['data']
    def list_outputs(self):
        return ['output']
    def list_auxiliary_states(self):
        return ["minmax"]
    def infer_shape(self, in_shape):
        shape = in_shape[0]
        if self.is_weight_perchannel and self.is_weight:
            aux_shape = [shape[0]]
        else:
            aux_shape = [1]
        return [shape], [shape], [aux_shape]
    def infer_type(self, in_type):
        return in_type, in_type, in_type 

    def create_operator(self, ctx, shapes, dtypes):
        return Quantization_int8(self.nbits, self.quant_mode, self.is_weight, self.is_weight_perchannel,
                                 self.delay_quant, self.ema_decay, self.grad_mode, self.fix_act_scale, self.bias_correct)



