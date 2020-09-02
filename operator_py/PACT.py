import mxnet as mx
import numpy as np
import copy


def print_info(auto_grad, cal_grad, name):
    print("{} autograd:\n{}\n cal grad:\n{}".format(name, auto_grad.asnumpy(), cal_grad.asnumpy()))
    print("{} autograd - cal_grad:\n{}".format(name, auto_grad.asnumpy() - cal_grad.asnumpy()))

def simulate_PACT(data, gamma, nbits):
    quant_level = 2**nbits - 1
    quant_unit = gamma / quant_level
    cliped_data = np.clip(data, 0, gamma)
    quantized_data = np.round(cliped_data/quant_unit) * quant_unit
    return quantized_data

def simulate_DoReFa(data, nbits):
    quant_level = 2**nbits - 1
    quant_unit = 1 / quant_level

    tanh_data = np.tanh(data)
    quantized_data = 2 * np.round( (tanh_data / (2 * np.max(np.abs(tanh_data))) + 0.5) / quant_unit ) * quant_unit - 1
    return quantized_data


def quantizeK(data, nbits):
    QUANT_LEVEL = 2**nbits - 1
    return mx.nd.round(QUANT_LEVEL * data) / QUANT_LEVEL

class DoReFa_PY(mx.operator.CustomOp):
    def __init__(self, nbits):
        self.nbits = nbits

        self.data = None
        self.output = None

    def forward(self, is_train, req, in_data, out_data, aux):
        # self.assign(out_data[0], req[0], in_data[0])
        # return

        # tanh_data = mx.nd.tanh(data)
        # max_abs = mx.nd.max(mx.nd.abs(tanh_data))

        self.data = in_data[0]
        self.data.attach_grad()
        with mx.autograd.record():
            tanh_data = mx.nd.tanh(self.data)
            v_max = mx.nd.max(mx.nd.abs(tanh_data))
            self.output = tanh_data / (2 * v_max) + 0.5
        self.assign(out_data[0], req[0], 2 * quantizeK(self.output, self.nbits) -1)
        
        
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        # self.assign(in_grad[0], req[0], out_grad[0])
        # return
        self.output.backward(2 * out_grad[0])
        self.assign(in_grad[0], req[0], self.data.grad)

        # data = in_data[0]
        # tanh_data = mx.nd.tanh(data)
        # # the diff of tanh(x)
        # diff_tanh_data = 1 - tanh_data * tanh_data
        
        # max_data = mx.nd.max(tanh_data)
        # max_abs_data = mx.nd.max(mx.nd.abs(tanh_data))
        # sign_flag = 2 * (max_abs_data ==  max_data) - 1
        # max_data = sign_flag * max_abs_data

        # '''
        # the diff of tanh(x)/2amx(|tanh(x)|)
        # 1/(2 * max_abs) if x!=max_abs;
        # -1/2 * sum(xi/max_abs**2), if x==max_abs and sign > 0
        # 1/2 * sum(xi/max_abs**2), if x==max_abs and sign < 0
        # '''
        # #### the sum should be the out_grad[0] * tanh_data, instead of sum(tanh_data)
        # diff_inter = 1 / (2 * max_abs_data) * out_grad[0] * (tanh_data != max_data) + \
        #              (- 1/ (2 * max_abs_data**2) ) * sign_flag * \
        #              mx.nd.sum(tanh_data * out_grad[0] * (tanh_data != max_data)) * (tanh_data == max_data)

        # data_grad = 2 * diff_tanh_data * diff_inter
        # print("sign_flag:{}, max_data:{}, max_abs:{}".format(sign_flag.asnumpy()[0], max_data.asnumpy()[0], 
        #                                                      max_abs_data.asnumpy()[0]))
        # print("out_grad:\n{}".format(out_grad[0]))
        # print("tanh data:\n{}".format(tanh_data))
        # print("auto grad:\n{}".format(self.data.grad))
        # print("cal grad:\n{}".format(data_grad))
        # print("auto - cal:\n{}".format(self.data.grad - data_grad))


@mx.operator.register("DoReFa_PY")
class DoReFa_PYProp(mx.operator.CustomOpProp):
    def __init__(self, nbits="8"):
        self.nbits = eval(nbits)
        super(DoReFa_PYProp, self).__init__(True)
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
        return DoReFa_PY(self.nbits)


class PACT_PY(mx.operator.CustomOp):
    def __init__(self, nbits):
        self.nbits = nbits
        self.QUANT_LEVEL = 2**self.nbits -1
        self.count=0

        self.data = None
        self.gamma = None
        self.output = None

    def forward(self, is_train, req, in_data, out_data, aux):
        # self.assign(out_data[0], req[0], in_data[0])
        # return
        assert len(in_data) == 2, "the input must be 2 in PACT: data and gamma"
        self.data = in_data[0]
        self.gamma = in_data[1]
        self.data.attach_grad()
        self.gamma.attach_grad()
        # print("{} gamma:{}".format(self.count, gamma.asnumpy()[0]))
        # self.count += 1
        # old_cliped = mx.nd.clip(self.data, 0, self.gamma.asnumpy()[0])

        with mx.autograd.record():
            self.output = mx.nd.where(self.data < self.gamma, self.data, self.gamma.broadcast_like(self.data))

        quant_unit = self.gamma / self.QUANT_LEVEL
        self.assign(out_data[0], req[0], mx.nd.round(self.output / quant_unit) * quant_unit)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        # self.assign(in_grad[0], req[0], out_grad[0])
        # return

        # cliped_flag = data >= gamma
        # # gamma_grad = 1 if data >= gamma
        # gamma_grad = mx.nd.sum(cliped_flag)
        # data = in_data[0]
        # gamma = in_data[1]
        # self.assign(in_grad[0], req[0], out_grad[0] * (data < gamma))
        # self.assign(in_grad[1], req[1], mx.nd.sum(out_grad[0] * (data >= gamma)))

        self.output.backward(out_grad[0])
        self.assign(in_grad[0], req[0], self.data.grad)
        self.assign(in_grad[1], req[1], self.gamma.grad)
        
@mx.operator.register("PACT_PY")
class PACT_PYProp(mx.operator.CustomOpProp):
    def __init__(self, nbits="8"):
        self.nbits = eval(nbits)
        super(PACT_PYProp, self).__init__(True)
    def list_arguments(self):
        return ['data', "gamma"]
    def list_outputs(self):
        return ['output']
    def list_auxiliary_states(self):
        return []
    def infer_shape(self, in_shape):
        shape = in_shape[0]
        return [shape, [1]], [shape], []
    def infer_type(self, in_type):
        dtype = in_type[0]
        return [dtype] * len(in_type), [in_type[0]]*len(self.list_outputs()), \
            [in_type[0]]*len(self.list_auxiliary_states())

    def create_operator(self, ctx, shapes, dtypes):
        return PACT_PY(self.nbits)


class PACT_V2_PY(mx.operator.CustomOp):
    def __init__(self, nbits):
        self.nbits = nbits
        self.QUANT_LEVEL = 2**(self.nbits) -1
        self.count=0

        self.data = None
        self.gamma = None
        self.output = None

    def forward(self, is_train, req, in_data, out_data, aux):
        # self.assign(out_data[0], req[0], in_data[0])
        # return
        assert len(in_data) == 2, "the input must be 2 in PACT: data and gamma"
        self.data = in_data[0]
        self.gamma = in_data[1]
        self.data.attach_grad()
        self.gamma.attach_grad()

        with mx.autograd.record():
            #  the below two implement is equal. and its' gradients is equal with autograd.

            data_sign = mx.nd.sign(self.data)
            data_abs = mx.nd.abs(self.data)
            self.output = mx.nd.where(data_abs < self.gamma, self.data, self.gamma.broadcast_like(self.data) * data_sign)
            # low_bound = mx.nd.where(self.data > - self.gamma, self.data, - self.gamma.broadcast_like(self.data))
            # self.output = mx.nd.where(low_bound < self.gamma, low_bound, self.gamma.broadcast_like(self.data))

        quant_unit = self.gamma / self.QUANT_LEVEL
        self.assign(out_data[0], req[0], mx.nd.round(self.output / quant_unit) * quant_unit)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.output.backward(out_grad[0])
        self.assign(in_grad[0], req[0], self.data.grad)
        self.assign(in_grad[1], req[1], self.gamma.grad)

        # # gamma_grad = 1 if data >= gamma
        # # gamma_grad = -1 if data <= -gamma
        # data = in_data[0]
        # gamma = in_data[1]
        # data_grad = out_grad[0] * (data < gamma) * (data > (-gamma))
        # gamma_grad = mx.nd.sum(out_grad[0] * (data >= gamma) - out_grad[0] * (data <= -gamma))
        
        # print_info(self.gamma.grad, gamma_grad, "gamma")

@mx.operator.register("PACT_V2_PY")
class PACT_V2_PYProp(mx.operator.CustomOpProp):
    def __init__(self, nbits="8"):
        self.nbits = eval(nbits)
        super(PACT_V2_PYProp, self).__init__(True)
    def list_arguments(self):
        return ['data', "gamma"]
    def list_outputs(self):
        return ['output']
    def list_auxiliary_states(self):
        return []
    def infer_shape(self, in_shape):
        shape = in_shape[0]
        return [shape, [1]], [shape], []
    def infer_type(self, in_type):
        dtype = in_type[0]
        return [dtype] * len(in_type), [in_type[0]]*len(self.list_outputs()), \
            [in_type[0]]*len(self.list_auxiliary_states())

    def create_operator(self, ctx, shapes, dtypes):
        return PACT_V2_PY(self.nbits)



class QUANT_STE_PY(mx.operator.CustomOp):
    def __init__(self, nbits):
        self.nbits = nbits
        self.QUANT_LEVEL = 2**(self.nbits-1) -1
        self.count=0


    def forward(self, is_train, req, in_data, out_data, aux):
        data = in_data[0]
        max_abs = mx.nd.max(mx.nd.abs(data))
        quant_unit = max_abs / self.QUANT_LEVEL
        self.assign(out_data[0], req[0], mx.nd.round(data / quant_unit) * quant_unit)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], out_grad[0])

@mx.operator.register("QUANT_STE_PY")
class QUANT_STE_PYProp(mx.operator.CustomOpProp):
    def __init__(self, nbits="8"):
        self.nbits = eval(nbits)
        super(QUANT_STE_PYProp, self).__init__(True)
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
        return QUANT_STE_PY(self.nbits)