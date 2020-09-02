/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2019 by Contributors
 * \file DoReFa-inl.h
* \author Xiaotao Chen
 * paper link: https://arxiv.org/abs/1606.06160
*/

#ifndef MXNET_OPERATOR_DoReFa_INL_H_
#define MXNET_OPERATOR_DoReFa_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../operator_common.h"
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "../tensor/control_flow_op.h"
#include "../quantization/quantization_utils.h"

namespace mxnet {
namespace op {

namespace DoReFa_enum {
enum DoReFaOpInputs {kData};
enum DoReFaOpOutputs {kOut};
enum DoReFaOpResource {kTempSpace};
}  // namespace DoReFa_enum


template <typename xpu, typename DType>
void print_data(mshadow::Tensor<xpu, 4, DType> data, mshadow::Stream<xpu> *s, const OpContext &ctx, std::string flag) {
    mshadow::Stream<cpu> *s_cpu = ctx.get_stream<cpu>();
    DType* temp;
    temp = (DType*) malloc(data.shape_.Size() * sizeof(DType));
    mshadow::Tensor<cpu, 4, DType> temp_tensor(temp, data.shape_, s_cpu);
    mshadow::Copy(temp_tensor, data, s);
    printf("--------------------------- %s ---------------------------\n", flag.c_str());
    for (int i=0; i< temp_tensor.size(0); i++) {
     for (int j=0; j< temp_tensor.size(1); j++) {
       for (int k=0; k< temp_tensor.size(2); k++) {
         for (int q=0; q< temp_tensor.size(3); q++) {
           printf("%f ", temp_tensor[i][j][k][q]);
         }
         printf("\n");
       }
       printf("\n");
     } 
     printf("\n");
    }
    printf("\n");
    free(temp);
}

template<typename xpu, typename DType>
DType get_scalar(mshadow::Tensor<xpu, 1, DType> data, mshadow::Stream<xpu> *s, const OpContext &ctx) {
    DType tmp_t = DType(0.0f);
    mshadow::Stream<cpu> *s_cpu = ctx.get_stream<cpu>();
    mshadow::Tensor<cpu, 1, DType> tmp_t_tensor(&tmp_t, mshadow::Shape1(1), s_cpu);
    mshadow::Copy(tmp_t_tensor, data, s);
    return tmp_t;
}

struct DoReFaPara : public dmlc::Parameter<DoReFaPara> {
  int nbits;
  DMLC_DECLARE_PARAMETER(DoReFaPara) {
    DMLC_DECLARE_FIELD(nbits).set_default(4)
    .describe("the target number of bits of quantization, default to 4.");
  }
};

template<typename xpu, typename DType>
class DoReFaOp : public Operator {
 public:
  explicit DoReFaOp(DoReFaPara param) {
    this->param_ = param;
    QUANT_LEVEL = std::pow(2, param.nbits) - 1;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 1U);
    CHECK_EQ(out_data.size(), 1U);
    CHECK_EQ(req.size(), 1U);
    CHECK_EQ(req[DoReFa_enum::kOut], kWriteTo);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4, DType> data;
    Tensor<xpu, 4, DType> out;
    if (in_data[DoReFa_enum::kData].ndim() == 2) {
      Shape<4> dshape = Shape4(in_data[DoReFa_enum::kData].shape_[0],
                               in_data[DoReFa_enum::kData].shape_[1], 1, 1);
      data = in_data[DoReFa_enum::kData].get_with_shape<xpu, 4, DType>(dshape, s);
      out = out_data[DoReFa_enum::kOut].get_with_shape<xpu, 4, DType>(dshape, s);
    } else {
      data = in_data[DoReFa_enum::kData].get<xpu, 4, DType>(s);
      out = out_data[DoReFa_enum::kOut].get<xpu, 4, DType>(s);
    }

    // allocate temp space
    mxnet::TShape src_shape, dst_shape;
    size_t temp_reduce_size = ConfigReduce<xpu, DType>(
        s, data.shape_, mxnet::TShape(1,1), &src_shape, &dst_shape);
    // space for tanh(data), abs(tanh(data)), temp_reduce_size, max(abs(tanh(data)))
    Tensor<xpu, 1, uint8_t> workspace = ctx.requested[DoReFa_enum::kTempSpace]
      .get_space_typed<xpu, 1, uint8_t>(Shape1(data.shape_.Size() * sizeof(DType) * 2 + temp_reduce_size + sizeof(DType)), s);
    uint64_t allocated_bytes = 0ULL;

    Tensor<xpu, 4, DType> tanh_data(reinterpret_cast<DType*>(workspace.dptr_ + allocated_bytes), data.shape_, s);
    allocated_bytes += tanh_data.shape_.Size() * sizeof(DType);
    Tensor<xpu, 4, DType> abs_tanh_data(reinterpret_cast<DType*>(workspace.dptr_ + allocated_bytes), data.shape_, s);
    allocated_bytes += abs_tanh_data.shape_.Size() * sizeof(DType);
    const int dev_id = ctx.run_ctx.ctx.dev_id;
    TBlob abs_tanh_data_t(reinterpret_cast<DType*>(abs_tanh_data.dptr_), abs_tanh_data.shape_, xpu::kDevMask, dev_id);
    Tensor<xpu, 1, char> temp_reduce_space(reinterpret_cast<char*>(workspace.dptr_ + allocated_bytes), 
                                           Shape1(temp_reduce_size), s);
    allocated_bytes += temp_reduce_size;
    Tensor<xpu, 1, DType> max_abs(reinterpret_cast<DType*>(workspace.dptr_ + allocated_bytes), Shape1(1), s);
    TBlob max_abs_t(reinterpret_cast<DType*>(max_abs.dptr_), max_abs.shape_, xpu::kDevMask, dev_id);
    allocated_bytes += sizeof(DType);

    tanh_data = F<mshadow_op::tanh>(data);
    abs_tanh_data = F<mshadow_op::abs>(tanh_data);

    // max(abs(tanh_data))
    broadcast::Reduce<red::maximum, 2, DType, mshadow::op::identity>(
        s, max_abs_t.reshape(dst_shape), kWriteTo, temp_reduce_space, abs_tanh_data_t.reshape(src_shape));
    
    // reuse the space of abs_tanh_data
    Tensor<xpu, 4, DType> tmp(reinterpret_cast<DType*>(abs_tanh_data.dptr_), data.shape_, s);
    ScalarExp<DType> half(0.5f);
    ScalarExp<DType> two(2.0f);
    tmp = tanh_data / broadcast_scalar(two * max_abs, data.shape_) + half;
    // to quantize 
    // reuse the space of max_abs
    Tensor<xpu, 1, DType> quant_unit(reinterpret_cast<DType*>(max_abs.dptr_), Shape1(1), s);
    // quant_unit = max_abs * ScalarExp<DType>(1.0 / QUANT_LEVEL);
    quant_unit = ScalarExp<DType>(1.0 / QUANT_LEVEL);
    tmp = F<mshadow_op::round>(tmp / broadcast_scalar(quant_unit, tmp.shape_)) * broadcast_scalar(quant_unit, tmp.shape_);
    Assign(out, req[DoReFa_enum::kOut], two * tmp - ScalarExp<DType>(1.0f));
  }
  

  virtual void Backward(const OpContext & ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    using namespace mxnet_op;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4, DType> data_grad;
    Tensor<xpu, 4, DType> out_data_grad;
    Tensor<xpu, 4, DType> data;
    Tensor<xpu, 4, DType> out;
    if (out_grad[DoReFa_enum::kOut].ndim() == 2) {
      Shape<4> dshape = Shape4(out_grad[DoReFa_enum::kOut].shape_[0],
                               out_grad[DoReFa_enum::kOut].shape_[1], 1, 1);
      data = in_data[DoReFa_enum::kData].get_with_shape<xpu, 4, DType>(dshape, s);
      out = out_data[DoReFa_enum::kOut].get_with_shape<xpu, 4, DType>(dshape, s);

      out_data_grad = out_grad[DoReFa_enum::kOut].get_with_shape<xpu, 4, DType>(dshape, s);
      data_grad = in_grad[DoReFa_enum::kData].get_with_shape<xpu, 4, DType>(dshape, s);
    } else {
      data = in_data[DoReFa_enum::kData].get<xpu, 4, DType>(s);
      out = out_data[DoReFa_enum::kOut].get<xpu, 4, DType>(s);

      out_data_grad = out_grad[DoReFa_enum::kOut].get<xpu, 4, DType>(s);
      data_grad = in_grad[DoReFa_enum::kData].get<xpu, 4, DType>(s);
    }

    /*
       the equation of DoReFa:
        tmp1 = tanh(data) / (2 * max(|tanh(data)|)) + 0.5  (1)
        tmp2 = round(tmp1 / quant_unit) * quant_unit       (2)
        tmp3 = 2 * tmp2 - 1                                (3)
       the diff of DoReFa:
        1. diff of tanh: 1 - tanh(data) * tanh(data)
        2. diff of x / (2 * max(|x|)) + 0.5:
            1 / (2 * max(|x|)), if x!=max(|x|),
            - 1 / 2 * sum(x_i / max(|x|)**2), if x==max(|x|) and max_x_i > 0,
            1 / 2 * sum(x_i / max(|x|)**2), if x==max(|x|) and max_x_i < 0,
        3. diff of (2): ste
        4. diff of (3): 2
    */
    
    // allocate temp space
    mxnet::TShape src_shape, dst_shape;
    size_t temp_reduce_size = ConfigReduce<xpu, DType>(
        s, data.shape_, mxnet::TShape(1,1), &src_shape, &dst_shape);
    // space for tanh(data), abs(tanh(data)), temp_reduce_size, max(tanh(data)), 
    // max(abs(tanh(data))), sum(tanh(data) * out_grad), sign_flag
    Tensor<xpu, 1, uint8_t> workspace = ctx.requested[DoReFa_enum::kTempSpace]
      .get_space_typed<xpu, 1, uint8_t>(Shape1(data.shape_.Size() * sizeof(DType) * 2 + temp_reduce_size + 
                                               sizeof(DType) * 4), s);
    uint64_t allocated_bytes = 0ULL;
    const int dev_id = ctx.run_ctx.ctx.dev_id;

    Tensor<xpu, 4, DType> tanh_data(reinterpret_cast<DType*>(workspace.dptr_ + allocated_bytes), data.shape_, s);
    allocated_bytes += tanh_data.shape_.Size() * sizeof(DType);
    TBlob tanh_data_t(reinterpret_cast<DType*>(tanh_data.dptr_), tanh_data.shape_, xpu::kDevMask, dev_id);
    Tensor<xpu, 4, DType> abs_tanh_data(reinterpret_cast<DType*>(workspace.dptr_ + allocated_bytes), data.shape_, s);
    allocated_bytes += abs_tanh_data.shape_.Size() * sizeof(DType);
    TBlob abs_tanh_data_t(reinterpret_cast<DType*>(abs_tanh_data.dptr_), abs_tanh_data.shape_, xpu::kDevMask, dev_id);
    Tensor<xpu, 1, char> temp_reduce_space(reinterpret_cast<char*>(workspace.dptr_ + allocated_bytes), 
                                           Shape1(temp_reduce_size), s);
    allocated_bytes += temp_reduce_size;
    Tensor<xpu, 1, DType> max_tanh_data(reinterpret_cast<DType*>(workspace.dptr_ + allocated_bytes), Shape1(1), s);
    TBlob max_t(reinterpret_cast<DType*>(max_tanh_data.dptr_), max_tanh_data.shape_, xpu::kDevMask, dev_id);
    allocated_bytes += sizeof(DType);
    Tensor<xpu, 1, DType> max_abs_tanh_data(reinterpret_cast<DType*>(workspace.dptr_ + allocated_bytes), Shape1(1), s);
    TBlob max_abs_t(reinterpret_cast<DType*>(max_abs_tanh_data.dptr_), max_abs_tanh_data.shape_, xpu::kDevMask, dev_id);
    allocated_bytes += sizeof(DType);
    Tensor<xpu, 1, DType> sum_tanh_grad(reinterpret_cast<DType*>(workspace.dptr_ + allocated_bytes), Shape1(1), s);
    TBlob sum_tanh_grad_t(reinterpret_cast<DType*>(sum_tanh_grad.dptr_), sum_tanh_grad.shape_, xpu::kDevMask, dev_id);
    allocated_bytes += sizeof(DType);
    Tensor<xpu, 1, DType> sign_flag(reinterpret_cast<DType*>(workspace.dptr_ + allocated_bytes), Shape1(1), s);
    allocated_bytes += sizeof(DType);


    tanh_data = F<mshadow_op::tanh>(data);
    abs_tanh_data = F<mshadow_op::abs>(tanh_data);
    // max(tanh_data)
    broadcast::Reduce<red::maximum, 2, DType, mshadow::op::identity>(
        s, max_t.reshape(dst_shape), kWriteTo, temp_reduce_space, tanh_data_t.reshape(src_shape));
    // max(abs(tanh_data))
    broadcast::Reduce<red::maximum, 2, DType, mshadow::op::identity>(
        s, max_abs_t.reshape(dst_shape), kWriteTo, temp_reduce_space, abs_tanh_data_t.reshape(src_shape));
    // sum(tanh_data * out_grad * (tanh_data != max_tanh_data))
    // reuse space of abs_tanh_data  
    Tensor<xpu, 4, DType> tanh_mul_grad(reinterpret_cast<DType*>(abs_tanh_data.dptr_), abs_tanh_data.shape_, s);
    TBlob tanh_mul_grad_t(reinterpret_cast<DType*>(tanh_mul_grad.dptr_), tanh_mul_grad.shape_, cpu::kDevMask, dev_id);
    tanh_mul_grad = tanh_data * out_data_grad * F<mshadow_op::ne>(tanh_data, broadcast_scalar(max_tanh_data, tanh_data.shape_));
    broadcast::Reduce<red::sum, 2, DType, mshadow::op::identity>(
        s, sum_tanh_grad_t.reshape(dst_shape), kWriteTo, temp_reduce_space, tanh_mul_grad_t.reshape(src_shape));

    ScalarExp<DType> two(2.0f);
    ScalarExp<DType> half(0.5f);
    ScalarExp<DType> m_half(-0.5f);
    ScalarExp<DType> one(1.0f);

    sign_flag = two * F<mshadow_op::eq>(max_tanh_data, max_abs_tanh_data) - one;
    max_tanh_data = sign_flag * max_abs_tanh_data;

    // reuse space of abs_tanh_data    
    Tensor<xpu, 4, DType> diff_2(reinterpret_cast<DType*>(abs_tanh_data.dptr_), abs_tanh_data.shape_, s);
    
    // sum(tanh_data * out_grad * (tanh_data != max_tanh_data)) instead of (sum(tanh_data) - max_tanh_data) * out_grad
    diff_2 = broadcast_scalar(half / max_abs_tanh_data, tanh_data.shape_) * out_data_grad * \
               (F<mshadow_op::ne>(tanh_data, broadcast_scalar(max_tanh_data, tanh_data.shape_))) + \
             broadcast_scalar(m_half / (max_abs_tanh_data * max_abs_tanh_data) * sign_flag * sum_tanh_grad, 
                              tanh_data.shape_) * \
               (F<mshadow_op::eq>(tanh_data, broadcast_scalar(max_tanh_data, tanh_data.shape_)));

    Assign(data_grad, req[DoReFa_enum::kOut], two * (one - tanh_data * tanh_data) * diff_2);
  }

 private:
  DoReFaPara param_;
  int QUANT_LEVEL;

};  // class DoReFaOp

template<typename xpu>
Operator* CreateOp(DoReFaPara type, int dtype);

#if DMLC_USE_CXX11
class DoReFaProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;

    CHECK_EQ(in_shape->size(), 1U) << "Input:[data]";
    const mxnet::TShape &dshape = in_shape->at(0);
    if (!mxnet::ndim_is_known(dshape)) return false;
    out_shape->clear();
    out_shape->push_back(dshape);
    aux_shape->clear();
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_EQ(in_type->size(), 1U);
    int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "First input must have specified type";
    for (size_t i = 0; i < in_type->size(); ++i) {
      if ((*in_type)[i] == -1) {
        (*in_type)[i] = dtype;
      } else {
        UNIFORM_TYPE_CHECK((*in_type)[i], dtype, ListArguments()[i]);
      }
    }

    out_type->clear();
    out_type->push_back(dtype);
    aux_type->clear();
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new DoReFaProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "_contrib_DoReFa";
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[DoReFa_enum::kOut], 
            in_data[DoReFa_enum::kData], 
            out_data[DoReFa_enum::kOut]};
  }

  std::vector<std::string> ListArguments() const override {
    return {"data"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output"};
  }

  std::vector<std::string> ListAuxiliaryStates() const override {
    return {};
  }

  int NumOutputs() const override {
    return 1;
  }

  int NumVisibleOutputs() const override {
    return 1;
  }

  std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  std::vector<ResourceRequest> BackwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                           std::vector<int> *in_type) const override;

 private:
  DoReFaPara param_;
};
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_Qunatization_Int8_INL_H_

