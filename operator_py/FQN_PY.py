import mxnet as mx
import numpy as np
import copy

def simulate(data, nbits, is_perchannel):
    quant_level = 2**nbits -1
    if is_perchannel:
        channel = data.shape[0]
        mins = np.min(data, axis=(1,2,3))
        maxs = np.max(data, axis=(1,2,3))
        mins = mins.reshape((channel, 1, 1, 1))
        maxs = maxs.reshape((channel, 1, 1, 1))
    else:
        mins = np.min(data)
        maxs = np.max(data)
    quant_unit = (maxs - mins) / quant_level
    quant_data = np.round((data - mins) / quant_unit) * quant_unit + mins
    return quant_data

class FQN(mx.operator.CustomOp):
    def __init__(self, 
                 nbits,
                 is_perchannel):
        self.nbits = nbits
        self.is_perchannel = is_perchannel
        self.QUANT_LEVEL = 2**self.nbits -1
    def forward(self, is_train, req, in_data, out_data, aux):
        data = in_data[0]
        if self.is_perchannel:
            # save weight maxs
            if is_train > 0:
                reduce_axis = tuple([i for i in range(len(data.shape))])
                mins = mx.nd.min(data, axis=reduce_axis[1:])
                maxs = mx.nd.max(data, axis=reduce_axis[1:])
                aux[0][:] = mins
                aux[1][:] = maxs + 1e-6 # avoid values of some channel is 0.

            target_shape = (in_data[0].shape[0],) + (1,) * len(in_data[0].shape[1:])
            quant_unit = (aux[1] - aux[0]) / self.QUANT_LEVEL
            quant_unit = quant_unit.reshape(target_shape).broadcast_like(in_data[0])
            mins = aux[0].reshape(target_shape).broadcast_like(in_data[0])
            self.assign(out_data[0], req[0], mx.nd.round((in_data[0] - mins) / quant_unit) * quant_unit + mins)
        else:
            if is_train > 0:
                mins = mx.nd.min(data)
                maxs = mx.nd.max(data)
                aux[0][:] = mins
                aux[1][:] = maxs
            quant_unit = (aux[1] - aux[0]) / self.QUANT_LEVEL
            self.assign(out_data[0], req[0], mx.nd.round( (in_data[0] - aux[0]) / quant_unit) * quant_unit + aux[0])

        # simulated_data = simulate(data.asnumpy(), self.nbits, self.is_perchannel)
        # print("data:{}".format(data.asnumpy()))
        # print("aux0:{}, aux1:{}".format(aux[0].asnumpy(), aux[1].asnumpy()))
        # print("simulate:{}".format(simulated_data))
        # print("cal:{}".format(out_data[0].asnumpy()))
        # print("simulate - cal:{}".format(simulated_data - out_data[0].asnumpy()))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], out_grad[0])

@mx.operator.register("FQN_PY")
class FQNProp(mx.operator.CustomOpProp):
    def __init__(self, 
                 nbits,
                 is_perchannel=False):
        self.nbits = int(nbits)
        self.is_perchannel = eval(is_perchannel)
        
        super(FQNProp, self).__init__(True)
    def list_arguments(self):
        return ['data']
    def list_outputs(self):
        return ['output']
    def list_auxiliary_states(self):
        return ["min", "max"]
    def infer_shape(self, in_shape):
        shape = in_shape[0]
        if self.is_perchannel:
            aux_shape = [shape[0]]
        else:
            aux_shape = [1]
        return [shape], [shape], [aux_shape, aux_shape]
    def infer_type(self, in_type):
        dtype = in_type[0]
        return [dtype] * len(in_type), [in_type[0]]*len(self.list_outputs()), \
            [in_type[0]]*len(self.list_auxiliary_states())

    def create_operator(self, ctx, shapes, dtypes):
        return FQN(self.nbits, self.is_perchannel)