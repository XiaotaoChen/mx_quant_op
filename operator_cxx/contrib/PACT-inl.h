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
 * \file PACT-inl.h
 * paper link: https://arxiv.org/abs/1805.06085
* \author Xiaotao Chen
*/

#ifndef MXNET_OPERATOR_PACT_INL_H_
#define MXNET_OPERATOR_PACT_INL_H_

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

namespace PACT_enum {
enum PACTOpInputs {kData, kGamma};
enum PACTOpOutputs {kOut};
enum PACTOpResource {kTempSpace};
}  // namespace PACT_enum


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

struct PACTPara : public dmlc::Parameter<PACTPara> {
  int nbits;
  DMLC_DECLARE_PARAMETER(PACTPara) {
    DMLC_DECLARE_FIELD(nbits).set_default(4)
    .describe("the target number of bits of quantization, default to 4.");
  }
};

template<typename xpu, typename DType>
class PACTOp : public Operator {
 public:
  explicit PACTOp(PACTPara param) {
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
    CHECK_EQ(in_data.size(), 2U);
    CHECK_EQ(out_data.size(), 1U);
    CHECK_EQ(req.size(), 1U);
    CHECK_EQ(req[PACT_enum::kOut], kWriteTo);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4, DType> data;
    Tensor<xpu, 4, DType> out;
    if (in_data[PACT_enum::kData].ndim() == 2) {
      Shape<4> dshape = Shape4(in_data[PACT_enum::kData].shape_[0],
                               in_data[PACT_enum::kData].shape_[1], 1, 1);
      data = in_data[PACT_enum::kData].get_with_shape<xpu, 4, DType>(dshape, s);
      out = out_data[PACT_enum::kOut].get_with_shape<xpu, 4, DType>(dshape, s);
    } else {
      data = in_data[PACT_enum::kData].get<xpu, 4, DType>(s);
      out = out_data[PACT_enum::kOut].get<xpu, 4, DType>(s);
    }
    Tensor<xpu, 1, DType> gamma = in_data[PACT_enum::kGamma].get<xpu, 1, DType>(s);
    
    Tensor<xpu, 1, uint8_t> workspace = ctx.requested[PACT_enum::kTempSpace]
        .get_space_typed<xpu, 1, uint8_t>(Shape1(sizeof(DType)), s);
    uint64_t alloclated_bytes = 0ULL;
    Tensor<xpu, 1, DType> quant_unit(reinterpret_cast<DType*>(workspace.dptr_ + alloclated_bytes), Shape1(1), s);

    quant_unit = gamma * ScalarExp<DType>(1.0 / QUANT_LEVEL);

    Assign(out, req[PACT_enum::kOut], F<mshadow_op::round>( 
            F<mshadow_op::clip>(data, broadcast_scalar(gamma, data.shape_)) / 
              broadcast_scalar(quant_unit, data.shape_)) * broadcast_scalar(quant_unit, data.shape_) );
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
    if (out_grad[PACT_enum::kOut].ndim() == 2) {
      Shape<4> dshape = Shape4(out_grad[PACT_enum::kOut].shape_[0],
                               out_grad[PACT_enum::kOut].shape_[1], 1, 1);
      data = in_data[PACT_enum::kData].get_with_shape<xpu, 4, DType>(dshape, s);
      out = out_data[PACT_enum::kOut].get_with_shape<xpu, 4, DType>(dshape, s);

      out_data_grad = out_grad[PACT_enum::kOut].get_with_shape<xpu, 4, DType>(dshape, s);
      data_grad = in_grad[PACT_enum::kData].get_with_shape<xpu, 4, DType>(dshape, s);
    } else {
      data = in_data[PACT_enum::kData].get<xpu, 4, DType>(s);
      out = out_data[PACT_enum::kOut].get<xpu, 4, DType>(s);

      out_data_grad = out_grad[PACT_enum::kOut].get<xpu, 4, DType>(s);
      data_grad = in_grad[PACT_enum::kData].get<xpu, 4, DType>(s);
    }
    Tensor<xpu, 1, DType> gamma_grad = in_grad[PACT_enum::kGamma].get<xpu, 1, DType>(s);
    Tensor<xpu, 1, DType> gamma = in_data[PACT_enum::kGamma].get<xpu, 1, DType>(s);
    // calculate gradient of gamma
    mxnet::TShape src_shape, dst_shape;
    size_t temp_reduce_size = ConfigReduce<xpu, DType>(
        s, out_data_grad.shape_, mxnet::TShape(1, 1), &src_shape, &dst_shape);    
    // space for outer_grad, temp_reduce_space, sum_t
    Tensor<xpu, 1, uint8_t> workspace = ctx.requested[PACT_enum::kTempSpace]
        .get_space_typed<xpu, 1, uint8_t>(Shape1(out_data_grad.shape_.Size() * sizeof(DType) + 
                                          temp_reduce_size + sizeof(DType)), s);
    uint64_t allocated_bytes = 0ULL;
    Tensor<xpu, 4, DType> outer_grad(reinterpret_cast<DType*>(workspace.dptr_ + allocated_bytes),
                                          out_data_grad.shape_, s);
    allocated_bytes += outer_grad.shape_.Size() * sizeof(DType);
    Tensor<xpu, 1, char> temp_reduce_space(reinterpret_cast<char*>(workspace.dptr_ + allocated_bytes),
                                           Shape1(temp_reduce_size), s);
    allocated_bytes += temp_reduce_size;
    Tensor<xpu, 1, DType> sum_grad(reinterpret_cast<DType*>(workspace.dptr_ + allocated_bytes),
                                   Shape1(1), s);
    allocated_bytes += sizeof(DType);

    const int dev_id = ctx.run_ctx.ctx.dev_id;
    TBlob sum_t(reinterpret_cast<DType*>(sum_grad.dptr_), sum_grad.shape_, xpu::kDevMask, dev_id);

    outer_grad = out_data_grad * F<mshadow_op::gt>(data, broadcast_scalar(gamma, data.shape_));
    TBlob outer_grad_t(reinterpret_cast<DType*>(outer_grad.dptr_), outer_grad.shape_, xpu::kDevMask, dev_id);

    // sum(outer_grad)
    broadcast::Reduce<red::sum, 2, DType, mshadow::op::identity>(
        s, sum_t.reshape(dst_shape), kWriteTo, temp_reduce_space, outer_grad_t.reshape(src_shape));
    // assign gamm_grad
    mshadow::Copy(gamma_grad, sum_grad, s);
    // assign clipped grad to data grad.
    Assign(data_grad, req[PACT_enum::kOut], out_data_grad * F<mshadow_op::le>(data, broadcast_scalar(gamma, data.shape_)));
  }

 private:
  PACTPara param_;
  int QUANT_LEVEL;

};  // class PACTOp

template<typename xpu>
Operator* CreateOp(PACTPara type, int dtype);

#if DMLC_USE_CXX11
class PACTProp : public OperatorProperty {
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

    CHECK_EQ(in_shape->size(), 2U) << "Input:[data, gamma]";
    const mxnet::TShape &dshape = in_shape->at(0);
    if (!mxnet::ndim_is_known(dshape)) return false;
    in_shape->at(1) = mxnet::TShape(Shape1(1));
    out_shape->clear();
    out_shape->push_back(dshape);
    aux_shape->clear();
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_EQ(in_type->size(), 2U);
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
    auto ptr = new PACTProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "_contrib_PACT";
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[PACT_enum::kOut], 
            in_data[PACT_enum::kData], 
            out_data[PACT_enum::kOut]};
  }

  std::vector<std::string> ListArguments() const override {
    return {"data", "gamma"};
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
  PACTPara param_;
};
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_Qunatization_Int8_INL_H_

