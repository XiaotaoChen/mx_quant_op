import mxnet as mx
import numpy as np
import copy

def print_info(cal_data, sim_data):
    print("cla data:\n`{}\n simulated:\n{}\n cal - simulate:\n{}".format(cal_data, sim_data, cal_data - sim_data))

def simulate_GDRQ(data, nbits, group_size, is_weight):
    if is_weight:
        nbits -= 1
    if group_size == -1:
        data_abs = np.abs(data)
        mean = np.mean(data_abs)
        threshold = 2 * mean
        if is_weight is False:
            threshold = 0.5 + 0.01 * (0.5 - threshold)
        quant_unit = threshold / (2**nbits -1)
        clipped_data = np.clip(data, - threshold, threshold)
        quanted_data = np.round(clipped_data / quant_unit) * quant_unit
    else:
        if is_weight is False:
            new_data = np.swapaxes(data, 0, 1)
        else:
            new_data = data
        shape = new_data.shape
        reshaped_shape = (shape[0]//group_size, group_size) + shape[1:]
        reshaped_data = np.reshape(new_data, reshaped_shape)
        reshaped_data_abs = np.abs(reshaped_data)
        reshaped_data_sign = np.sign(reshaped_data)

        axises = tuple([i for i in range(len(reshaped_shape))])
        mean = np.mean(reshaped_data_abs, axis=axises[1:])
        
        target_shape = (mean.shape) + (1,) * len(reshaped_shape[1:])
        reshaped_mean = np.reshape(mean, target_shape)

        threshold =  2 * reshaped_mean
        if is_weight is False:
            threshold = 0.5 + 0.01 * (0.5 - threshold)


        quant_unit = threshold / (2**nbits -1)
        clipped_data = np.where(reshaped_data_abs <= threshold, reshaped_data, threshold * reshaped_data_sign)
        quanted_data = np.round(clipped_data / quant_unit) * quant_unit
        quanted_data = np.reshape(quanted_data, shape)
        if is_weight is False:
            quanted_data = np.swapaxes(quanted_data, 0, 1)
    return quanted_data

class GDRQ_PY(mx.operator.CustomOp):
    def __init__(self, nbits, group_size, is_weight, lamda):
        self.nbits = nbits
        self.group_size = group_size
        self.is_weight = is_weight
        self.lamda = lamda
        self.QUANT_LEVEL = 2**(self.nbits) - 1
        

        # assert self.is_weight_perchannel == False, "currently GDRQ only support per tensor quantization"
    def forward(self, is_train, req, in_data, out_data, aux):
        data = in_data[0]
        alpha = aux[0]
        data_abs = mx.nd.abs(data)

        if self.group_size == -1:
            mean = mx.nd.mean(data_abs)
            threshold = 3 * mean
            if self.is_weight:
                alpha[:] = threshold
            else:
                alpha[:] = alpha + self.lamda * (alpha - threshold)

            quant_unit = alpha / self.QUANT_LEVEL
            scalar_t = alpha.asnumpy()[0]
            clipped_data = mx.nd.clip(data, - scalar_t, scalar_t)
            quanted_data = mx.nd.round(clipped_data / quant_unit) * quant_unit
            self.assign(out_data[0], req[0], quanted_data)
        else:
            if self.is_weight is False:
                data = mx.nd.swapaxes(data, 0, 1)  # for activation the channels in shape[1]
            
            shape = data.shape
            reshaped_shape = (shape[0] // self.group_size, self.group_size) + shape[1:]
            reshaped_data = mx.nd.reshape(data, shape=reshaped_shape)
            reshaped_data_abs = mx.nd.abs(reshaped_data)
            reshaped_data_sign = mx.nd.sign(reshaped_data)

            axises = tuple([i for i in range(len(reshaped_shape))])
            mean = mx.nd.mean(reshaped_data_abs, axis=axises[1:])
            threshold = 3 * mean
            if self.is_weight:
                alpha[:] = threshold
            else:
                alpha[:] = alpha + self.lamda * (alpha - threshold)

            target_shape = alpha.shape + (1,) * len(reshaped_shape[1:])
            reshaped_alpha = mx.nd.reshape(alpha, shape=target_shape)

            quant_unit = reshaped_alpha / self.QUANT_LEVEL
            clipped_data = mx.nd.where(reshaped_data_abs <= reshaped_alpha, reshaped_data, reshaped_alpha * reshaped_data_sign)
            quanted_data = mx.nd.round(clipped_data / quant_unit) * quant_unit
            quanted_data = mx.nd.reshape(quanted_data, shape=shape)
            if self.is_weight is False:
                quanted_data = mx.nd.swapaxes(quanted_data, 0, 1)
            self.assign(out_data[0], req[0], quanted_data)
        
        # sim_quanted_data = simulate_GDRQ(in_data[0].asnumpy(), self.nbits, self.group_size, self.is_weight)
        # print_info(out_data[0].asnumpy(), sim_quanted_data)


    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        if self.is_weight:
            self.assign(in_grad[0], req[0], out_grad[0])
            return
        data = in_data[0]
        alpha = aux[0]
        o_grad = out_grad[0]
        if self.group_size == -1:
            clip_flag = mx.nd.abs(data) <= alpha
            self.assign(in_grad[0], req[0], out_grad[0] * clip_flag)
        else:
            if self.is_weight is False:
                o_grad = mx.nd.swapaxes(o_grad, 0,1)
                data = mx.nd.swapaxes(data, 0, 1)  # for activation the channels in shape[1]

            shape = data.shape
            reshaped_shape = (shape[0] // self.group_size, self.group_size) + shape[1:]
            reshaped_data_abs = mx.nd.abs(mx.nd.reshape(data, shape=reshaped_shape))

            target_shape = alpha.shape + (1,) * len(reshaped_shape[1:])
            reshaped_alpha = mx.nd.reshape(alpha, shape=target_shape)
            clip_flag = reshaped_data_abs <= reshaped_alpha

            reshaped_grad = mx.nd.reshape(o_grad, shape=reshaped_shape)
            clipped_grad = reshaped_grad * clip_flag
            clipped_grad = mx.nd.reshape(clipped_grad, shape=shape)
            if self.is_weight is False:
                clipped_grad = mx.nd.swapaxes(clipped_grad, 0, 1)
            self.assign(in_grad[0], req[0], clipped_grad)
        
@mx.operator.register("GDRQ_PY")
class GDRQ_PYProp(mx.operator.CustomOpProp):
    def __init__(self, nbits=4, group_size=-1, is_weight=False,  lamda=0.99):
        self.nbits = int(nbits)
        self.group_size = int(group_size)
        self.is_weight = eval(is_weight)
        self.lamda = float(lamda)
        super(GDRQ_PYProp, self).__init__(True)
    def list_arguments(self):
        return ['data']
    def list_outputs(self):
        return ['output']
    def list_auxiliary_states(self):
        return ["alpha"]
    def infer_shape(self, in_shape):
        shape = in_shape[0]
        if self.group_size == -1:
            aux_shape=[1]
        else:
            if self.is_weight:
                channels = shape[0]
            else:
                channels = shape[1]
            assert channels % self.group_size == 0, "the channels of weight or activation must be divisible \
                by group size. channels({}) vs group size({}).".format(channels, self.group_size)
            aux_shape = [channels // self.group_size]
        return [shape], [shape], [aux_shape]
    def infer_type(self, in_type):
        dtype = in_type[0]
        return [dtype] * len(in_type), [in_type[0]]*len(self.list_outputs()), \
            [in_type[0]]*len(self.list_auxiliary_states())

    def create_operator(self, ctx, shapes, dtypes):
        return GDRQ_PY(self.nbits, self.group_size, self.is_weight, self.lamda)

class CLIP_RELU_PY(mx.operator.CustomOp):
    def __init__(self, nbits, threshold):
        self.nbits = nbits
        self.threshold = threshold
        self.QUANT_LEVEL = 2**(self.nbits) -1
        self.count=0


    def forward(self, is_train, req, in_data, out_data, aux):
        data = in_data[0]
        clipped_data = mx.nd.clip(data, 0, self.threshold)
        quant_unit = self.threshold / self.QUANT_LEVEL
        self.assign(out_data[0], req[0], mx.nd.round(clipped_data / quant_unit) * quant_unit)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        clipped_flag = in_data[0] < self.threshold
        self.assign(in_grad[0], req[0], out_grad[0] * clipped_flag)

@mx.operator.register("CLIP_RELU_PY")
class CLIP_RELU_PYProp(mx.operator.CustomOpProp):
    def __init__(self, nbits="8", threshold="8.0"):
        self.nbits = eval(nbits)
        self.threshold = eval(threshold)
        super(CLIP_RELU_PYProp, self).__init__(True)
    def list_arguments(self):
        return ['data']
    def list_outputs(self):
        return ['output']
    def list_auxiliary_states(self):
        return []
    def infer_shape(self, in_shape):
        shape = in_shape[0]
        return [shape], [shape], []
    def infer_type(self, in_type):
        dtype = in_type[0]
        return [dtype] * len(in_type), [in_type[0]]*len(self.list_outputs()), \
            [in_type[0]]*len(self.list_auxiliary_states())

    def create_operator(self, ctx, shapes, dtypes):
        return CLIP_RELU_PY(self.nbits, self.threshold)