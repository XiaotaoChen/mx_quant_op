import mxnet as mx
import numpy as np
import copy


def simulate_QIL_BW(data, pruning_point, clipping_point, out_grad):
    print("clip point:{}, pruning point:{}".format(clipping_point, pruning_point))
    data_abs = np.abs(data)
    data_sign = np.sign(data)
    interval_out_grad = (data_abs > pruning_point) * (data_abs < clipping_point) * out_grad
    data_grad = interval_out_grad / (clipping_point - pruning_point)
    pruning_point_grad = np.sum( interval_out_grad * ( (data - clipping_point * data_sign) / ((clipping_point - pruning_point)**2) ) )
    clipping_point_grad = np.sum(interval_out_grad * ( - (data - pruning_point * data_sign) / ((clipping_point - pruning_point)**2) ) )

def assert_all(pruning_point, clipping_point):
    pruning = pruning_point.asnumpy()
    clipping = clipping_point.asnumpy()
    # if np.all(pruning >=0) is False:
    #     pruning_point = mx.nd.zeros_like(pruning_point)
    assert np.all(pruning >= 0), "pruning {} must greater than 0".format(pruning[0])
    assert np.all(pruning < clipping), "pruning vs clipping {} vs {} pruning must less \
        than clipping".format(pruning[0], clipping[0])
    assert np.all(clipping <= 1), "clipping {} must less than 1.0".format(clipping[0])

def interval_quantize_signed(interval_data, sign_data, pruning_point, clipping_point, quant_level):
    interval = (clipping_point - pruning_point) / quant_level
    return mx.nd.round( (interval_data - pruning_point * sign_data) / interval ) * interval + pruning_point * sign_data

def interval_quantize(interval_data, pruning_point, clipping_point, quant_level):
    interval = (clipping_point - pruning_point) / quant_level
    return mx.nd.round( (interval_data - pruning_point) / interval ) * interval + pruning_point


class QIL_PY(mx.operator.CustomOp):
    def __init__(self, is_weight, fix_gamma, nbits):
        self.is_weight = is_weight
        self.fix_gamma = fix_gamma
        self.nbits = nbits
        self.QUANT_LEVEL = 2**self.nbits -1


        self.data = None
        self.pruning_point = None
        self.clipping_point = None
        self.output = None

        self.count = 0
        self.quantized_type = "ste"

    def forward(self, is_train, req, in_data, out_data, aux):
        if in_data[1].asnumpy()[0] < 0:
            in_data[1][:] = 0.0
        if in_data[2].asnumpy()[0] > 1.0:
            in_data[2][:] = 1.0

        assert in_data[1].asnumpy()[0] < in_data[2].asnumpy()[0], "pruning_point vs clipping_point, {} vs {}".format( \
                                                                  in_data[1].asnumpy()[0], in_data[2].asnumpy()[0])

        self.data = in_data[0]
        self.pruning_point = in_data[1]
        self.clipping_point = in_data[2]
        gamma = in_data[3]

        self.data.attach_grad()
        self.pruning_point.attach_grad()
        self.clipping_point.attach_grad()
        
        # print("count:{}, pruning:{}, clippig:{}".format(self.count, self.pruning_point.asnumpy()[0], self.clipping_point.asnumpy()[0]))
        # self.count += 1

        with mx.autograd.record():
            center = 0.5 * (self.clipping_point + self.pruning_point)
            distance = 0.5 * (self.clipping_point - self.pruning_point)
            alpha = 0.5 / distance
            beta =  -0.5 * center / distance + 0.5
            data_abs = mx.nd.abs(self.data)
            data_sign = mx.nd.sign(self.data)
            interval_flag = (data_abs >= self.pruning_point) * (data_abs <= self.clipping_point)
            self.output = data_sign * (data_abs > self.clipping_point) + \
                          data_sign * (alpha * data_abs + beta) * interval_flag
        if self.quantized_type == "interval":
            self.assign(out_data[0], req[0], interval_quantize(self.output * data_sign, self.pruning_point,
                                                               self.clipping_point, self.QUANT_LEVEL) * data_sign)
        else:
            self.assign(out_data[0], req[0], mx.nd.round(self.output * self.QUANT_LEVEL) / self.QUANT_LEVEL)
        
        # pruning = pruning_point.asnumpy()
        # clipping = clipping_point.asnumpy()
        # if pruning[0] < 0:
        #     in_data[1][:] = mx.nd.zeros_like(in_data[1])[:]
        #     pruning_point = in_data[1]
        # if clipping[0] > 1:
        #     in_data[2][:] = mx.nd.ones_like(in_data[2])[:]
        #     clipping_point = in_data[2]

        # center = 0.5 * (clipping_point + pruning_point)
        # distance = 0.5 * (clipping_point - pruning_point)
        # alpha = 0.5 / distance
        # beta = -0.5 * center / distance + 0.5

        # data_abs = mx.nd.abs(data)
        # data_sign = mx.nd.sign(data)
        # interval_data = data_abs * (data_abs > pruning_point) * (data_abs < clipping_point)
        # if self.is_weight:
        #     transformed_data = data_sign * (data_abs > clipping_point) + \
        #                        data_sign * (alpha * interval_data + beta)
        #     # transformed_data = data_sign * (data_abs > clipping_point) + \
        #     #                    data_sign * mx.nd.power((alpha * interval_data + beta), gamma)
        #     self.assign(out_data[0], req[0], (mx.nd.round(transformed_data * self.QUANT_LEVEL) / 
        #                                           self.QUANT_LEVEL) )
        # else:
        #     transformed_data = data_sign * (data_abs > clipping_point) + \
        #                        data_sign * (alpha * interval_data + beta)
        #     self.assign(out_data[0], req[0], (mx.nd.round(transformed_data * self.QUANT_LEVEL) / 
        #                                          self.QUANT_LEVEL) )

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        assert len(req) >= 3
        assert self.fix_gamma == True, "currently only support fix gamma"

        self.output.backward(out_grad[0])
        self.assign(in_grad[0], req[0], self.data.grad)
        self.assign(in_grad[1], req[1], self.pruning_point.grad)
        self.assign(in_grad[2], req[2], self.clipping_point.grad)

        # data = in_data[0]
        # pruning_point = in_data[1]
        # clipping_point = in_data[2]
        # gamma = in_data[3]

        # data_abs = mx.nd.abs(data)
        # data_sign = mx.nd.sign(data)
        # interval_flag = (data_abs >= pruning_point) * (data_abs <= clipping_point)
        
        # data_grad = out_grad[0] / (clipping_point - pruning_point) * interval_flag

        # pruning_grad = mx.nd.sum(out_grad[0] * ( (data - clipping_point* data_sign) /
        #                                           ((clipping_point - pruning_point)**2) ) * interval_flag )
        # clipping_grad = mx.nd.sum(out_grad[0] * (- (data - pruning_point * data_sign) / 
        #                                             ((clipping_point - pruning_point)**2) ) * interval_flag )

        # print("data:\n{}\n pruning_point:{}, clipping_point:{}".format(data, pruning_point.asnumpy()[0], clipping_point.asnumpy()[0]))
        # print("out grad:\n{}\nsign:\n{}\n interval flag:\n{}".format(out_grad[0], data_sign, interval_flag))


        # print("data.grad:\n{}\n cal data grad:\n{}".format(self.data.grad, data_grad))
        # print("data.grad - cal_grad:\n{}".format(self.data.grad - data_grad))

        # print("pruning_point.grad:\n{}\n cal pruning grad:\n{}".format(self.pruning_point.grad, pruning_grad))
        # print("pruning_point.grad - cal_grad:\n{}".format(self.pruning_point.grad - pruning_grad))
        # print("clipping_point.grad:\n{}\n cal clipping grad:\n{}".format(self.clipping_point.grad, clipping_grad))
        # print("clipping_point.grad - cal_grad:\n{}".format(self.clipping_point.grad - clipping_grad))
        
@mx.operator.register("QIL_PY")
class QIL_PYProp(mx.operator.CustomOpProp):
    def __init__(self, is_weight="False", fix_gamma="True", nbits="4"):
        self.is_weight = eval(is_weight)
        self.fix_gamma = eval(fix_gamma)
        self.nbits = int(nbits)
        super(QIL_PYProp, self).__init__(True)
    def list_arguments(self):
        return ['data', 'pruning_point', 'clipping_point', 'gamma']
    def list_outputs(self):
        return ['output']
    def list_auxiliary_states(self):
        return []
    def infer_shape(self, in_shape):
        shape = in_shape[0]
        return [shape, [1], [1], [1]], [shape], []
    def infer_type(self, in_type):
        dtype = in_type[0]
        return [dtype] * len(in_type), [in_type[0]]*len(self.list_outputs()), \
            [in_type[0]]*len(self.list_auxiliary_states())

    def create_operator(self, ctx, shapes, dtypes):
        return QIL_PY(self.is_weight, self.fix_gamma, self.nbits)
