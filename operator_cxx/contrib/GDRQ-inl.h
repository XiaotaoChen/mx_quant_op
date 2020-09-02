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
 * \file GDRQ-inl.h
 * paper link: https://arxiv.org/abs/1908.01477
* \author Xiaotao Chen
*/

#ifndef MXNET_OPERATOR_GDRQ_INL_H_
#define MXNET_OPERATOR_GDRQ_INL_H_

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

namespace GDRQ_enum {
enum GDRQOpInputs {kData};
enum GDRQOpOutputs {kOut};
enum GDRQOpAuxiliary {kAlpha};
enum GDRQOpResource {kTempSpace};
}  // namespace GDRQ_enum


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

struct GDRQPara : public dmlc::Parameter<GDRQPara> {
  int nbits;
  int group_size;
  bool is_weight;
  float lamda;
  bool do_quant;
  bool fix_alpha;
  float ktimes;
  std::string grad_mode;
  DMLC_DECLARE_PARAMETER(GDRQPara) {
    DMLC_DECLARE_FIELD(nbits).set_default(4)
    .describe("the target number of bits of quantization, default to 4.");
    DMLC_DECLARE_FIELD(group_size).set_default(-1)
    .describe("the group size for quantization, default to -1, which means per layer quantization.");
    DMLC_DECLARE_FIELD(is_weight).set_default(false)
    .describe("the Tensor is weight or not.");
    DMLC_DECLARE_FIELD(lamda).set_default(0.001)
    .describe("the coefficient for update the clipping threshold on Activation, defalut to 0.001");
    DMLC_DECLARE_FIELD(do_quant).set_default(false)
    .describe("do quantize or not, default to false.");
    DMLC_DECLARE_FIELD(fix_alpha).set_default(false)
    .describe("fix alpha or not for activation, default to false.");
    DMLC_DECLARE_FIELD(ktimes).set_default(2)
    .describe("the coefficient to calculate threshold with mean. threshold = ktimes * mean. defalut to 2.");
    DMLC_DECLARE_FIELD(grad_mode).set_default("ste")
    .describe("the gradients passing mode: ste or clip. defalut to ste");
  }
};

template<typename xpu, typename DType>
class GDRQOp : public Operator {
 public:
  explicit GDRQOp(GDRQPara param) {
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
    CHECK_EQ(req[GDRQ_enum::kOut], kWriteTo);
    CHECK_EQ(param_.group_size, -1) << "currently only support per layer quantization. which means group_size = -1";

    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4, DType> data;
    Tensor<xpu, 4, DType> out;
    if (in_data[GDRQ_enum::kData].ndim() == 2) {
      Shape<4> dshape = Shape4(in_data[GDRQ_enum::kData].shape_[0],
                               in_data[GDRQ_enum::kData].shape_[1], 1, 1);
      data = in_data[GDRQ_enum::kData].get_with_shape<xpu, 4, DType>(dshape, s);
      out = out_data[GDRQ_enum::kOut].get_with_shape<xpu, 4, DType>(dshape, s);
    } else {
      data = in_data[GDRQ_enum::kData].get<xpu, 4, DType>(s);
      out = out_data[GDRQ_enum::kOut].get<xpu, 4, DType>(s);
    }
    Tensor<xpu, 1, DType> aux = aux_states[GDRQ_enum::kAlpha].get<xpu, 1, DType>(s);
    // calculate temp space
    mxnet::TShape src_shape, dst_shape;
    size_t temp_reduce_size;
    Tensor<xpu, 1, uint8_t> workspace;
    if (!param_.fix_alpha) {
      temp_reduce_size = ConfigReduce<xpu, DType>(
        s, data.shape_, mxnet::TShape(1, 1), &src_shape, &dst_shape);
      // space for tmp_scalar, tmp_data/data_abs, temp_reudce_size
      workspace = ctx.requested[GDRQ_enum::kTempSpace]
        .get_space_typed<xpu, 1, uint8_t>(Shape1(sizeof(DType) + data.shape_.Size() * sizeof(DType) + 
                                                 temp_reduce_size), s);
    }
    else {
      // space for tmp_scalar, tmp_data
      workspace = ctx.requested[GDRQ_enum::kTempSpace]
        .get_space_typed<xpu, 1, uint8_t>(Shape1(sizeof(DType) + data.shape_.Size() * sizeof(DType)), s);
    }
    uint64_t allocated_bytes = 0ULL;
    Tensor<xpu, 1, DType> tmp_scalar(reinterpret_cast<DType*>(workspace.dptr_ + allocated_bytes), Shape1(1), s);
    allocated_bytes += sizeof(DType);

    Tensor<xpu, 4, DType> tmp_data(reinterpret_cast<DType*>(workspace.dptr_ + allocated_bytes), data.shape_, s);
    allocated_bytes += tmp_data.shape_.Size() * sizeof(DType);

    if (!param_.fix_alpha) {
      const int dev_id = ctx.run_ctx.ctx.dev_id;
      TBlob data_abs_t(reinterpret_cast<DType *>(tmp_data.dptr_), data.shape_, xpu::kDevMask, dev_id);

      tmp_data = F<mshadow_op::abs>(data);
      
      Tensor<xpu, 1, char> temp_reduce_space(reinterpret_cast<char*>(workspace.dptr_ + allocated_bytes), 
                                             Shape1(temp_reduce_size), s);
      allocated_bytes += temp_reduce_size;
      
      TBlob sum_t(reinterpret_cast<DType *>(tmp_scalar.dptr_), Shape1(1), xpu::kDevMask, dev_id);

      // sum(data_abs)
      broadcast::Reduce<red::sum, 2, DType, mshadow::op::identity>(
        s, sum_t.reshape(dst_shape), kWriteTo, temp_reduce_space, data_abs_t.reshape(src_shape));
      // mean
      ScalarExp<DType> ktimes_expr(DType(param_.ktimes));
      real_t total_num = static_cast<real_t>(data.shape_.Size());
      ScalarExp<DType> total_num_expr(DType(1.0 / total_num));
      tmp_scalar = ktimes_expr * (tmp_scalar * total_num_expr);

      if (param_.is_weight) {
        mshadow::Copy(aux, tmp_scalar, s);
      }
      else {
        ScalarExp<DType> lamda_expr(param_.lamda);
        aux = aux + lamda_expr * (aux - tmp_scalar);
      }
    }

    tmp_data = F<mshadow_op::clip>(data, broadcast_scalar(aux, data.shape_));

    // assign data to out
    if (ctx.is_train > 0 && !param_.do_quant) {
      mshadow::Copy(out, tmp_data, s);
    }
    else {
      ScalarExp<DType> quant_level_expr(DType(1.0 / QUANT_LEVEL));
      tmp_scalar = aux * quant_level_expr;
      Assign(out, req[GDRQ_enum::kOut], F<mshadow_op::round>(tmp_data / broadcast_scalar(tmp_scalar, tmp_data.shape_)) * broadcast_scalar(tmp_scalar, tmp_data.shape_));
    }
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
    if (out_grad[GDRQ_enum::kOut].ndim() == 2) {
      Shape<4> dshape = Shape4(out_grad[GDRQ_enum::kOut].shape_[0],
                               out_grad[GDRQ_enum::kOut].shape_[1], 1, 1);
      data = in_data[GDRQ_enum::kData].get_with_shape<xpu, 4, DType>(dshape, s);
      out = out_data[GDRQ_enum::kOut].get_with_shape<xpu, 4, DType>(dshape, s);

      out_data_grad = out_grad[GDRQ_enum::kOut].get_with_shape<xpu, 4, DType>(dshape, s);
      data_grad = in_grad[GDRQ_enum::kData].get_with_shape<xpu, 4, DType>(dshape, s);
    } else {
      data = in_data[GDRQ_enum::kData].get<xpu, 4, DType>(s);
      out = out_data[GDRQ_enum::kOut].get<xpu, 4, DType>(s);

      out_data_grad = out_grad[GDRQ_enum::kOut].get<xpu, 4, DType>(s);
      data_grad = in_grad[GDRQ_enum::kData].get<xpu, 4, DType>(s);
    }

    if (param_.is_weight && param_.grad_mode == std::string("ste")) {
      mshadow::Copy(data_grad, out_data_grad, s);
    }
    else {
      Tensor<xpu, 1, DType> aux = aux_states[GDRQ_enum::kAlpha].get<xpu, 1, DType>(s);
      // Assign(data_grad, req[GDRQ_enum::kOut], out_data_grad * F<mshadow_op::le>(data, broadcast_scalar(aux, data.shape_)));
      
      Tensor<xpu, 1, uint8_t> workspace = ctx.requested[GDRQ_enum::kTempSpace]
          .get_space_typed<xpu, 1, uint8_t>(Shape1(sizeof(DType)), s);
      Tensor<xpu, 1, DType> minus_aux(reinterpret_cast<DType*>(workspace.dptr_), Shape1(1), s);
      DType aux_num = get_scalar<xpu, DType>(aux, s, ctx);
      ScalarExp<DType> tmp_t_expr(- aux_num);
      minus_aux = tmp_t_expr;

      Assign(data_grad, req[GDRQ_enum::kOut], out_data_grad * 
                                              F<mshadow_op::le>(data, broadcast_scalar(aux, data.shape_) * 
                                              F<mshadow_op::ge>(data, broadcast_scalar(minus_aux, data.shape_))));
    }
  }

 private:
  GDRQPara param_;
  int QUANT_LEVEL;

};  // class GDRQOp

template<typename xpu>
Operator* CreateOp(GDRQPara type, int dtype);

#if DMLC_USE_CXX11
class GDRQProp : public OperatorProperty {
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
    const TShape &dshape = in_shape->at(GDRQ_enum::kData);
    out_shape->clear();
    out_shape->push_back(dshape);
    
    Shape<1>  tmp_aux_shape = Shape1(1);
    if (param_.group_size != -1) {
      if (param_.is_weight) {
        CHECK_EQ(dshape[0] % param_.group_size, 0) << "the number of filters for weight must be divided \
                                                       by group_size: "<< param_.group_size;
        tmp_aux_shape = Shape1(dshape[0] / param_.group_size);
      }
      else {
        CHECK_EQ(dshape[1] % param_.group_size, 0) << "the number of channels for input must be divided \
                                                       by group_size: "<< param_.group_size;
        tmp_aux_shape = Shape1(dshape[1] / param_.group_size);
      }
    }
    aux_shape->clear();
    aux_shape->push_back(TShape(tmp_aux_shape));
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
    aux_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new GDRQProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "_contrib_GDRQ";
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[GDRQ_enum::kOut], 
            in_data[GDRQ_enum::kData], 
            out_data[GDRQ_enum::kOut]};
  }

  std::vector<std::string> ListArguments() const override {
    return {"data"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output"};
  }

  std::vector<std::string> ListAuxiliaryStates() const override {
    return {"alpha"};
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
  GDRQPara param_;
};
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_Qunatization_Int8_INL_H_

