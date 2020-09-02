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
 * \file quantization_int8-inl.h
 * paper link: http://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Fully_Quantized_Network_for_Object_Detection_CVPR_2019_paper.pdf
* \author Xiaotao Chen
*/

#ifndef MXNET_OPERATOR_FQN_INL_H_
#define MXNET_OPERATOR_FQN_INL_H_

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
#include "../tensor/indexing_op.h"
#include "../quantization/quantization_utils.h"

namespace mxnet {
namespace op {

namespace FQN_enum {
enum FQNOpInputs {kData};
enum FQNOpOutputs {kOut};
enum FQNOpAuxiliary {kMin, kMax};
enum FQNOpResource {kTempSpace};
}  // namespace FQN_enum


template <typename xpu, typename DType>
void print_data_1D(mshadow::Tensor<xpu, 1, DType> data, mshadow::Stream<xpu> *s, const OpContext &ctx, std::string flag) {
    mshadow::Stream<cpu> *s_cpu = ctx.get_stream<cpu>();
    DType* temp;
    temp = (DType*) malloc(data.shape_.Size() * sizeof(DType));
    mshadow::Tensor<cpu, 1, DType> temp_tensor(temp, data.shape_, s_cpu);
    mshadow::Copy(temp_tensor, data, s);
    printf("--------------------------- %s ---------------------------\n", flag.c_str());
    for (int i=0; i< temp_tensor.size(0); i++) {
      printf("%f ", temp_tensor[i]);
    }
    printf("\n");
    free(temp);
}

template <typename xpu, typename DType>
void print_data_4D(mshadow::Tensor<xpu, 4, DType> data, mshadow::Stream<xpu> *s, const OpContext &ctx, std::string flag) {
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


struct FQNPara : public dmlc::Parameter<FQNPara> {
  int nbits;
  bool is_perchannel;
  DMLC_DECLARE_PARAMETER(FQNPara) {
    DMLC_DECLARE_FIELD(nbits).set_default(4)
    .describe("the target number of bits of quantization, default to 4.");
    DMLC_DECLARE_FIELD(is_perchannel).set_default(false)
    .describe("if true, this quantization layer is used with per channel quantize");
  }
};

template<typename xpu, typename DType>
class FQNOp : public Operator {
 public:
  explicit FQNOp(FQNPara param) {
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
    CHECK_EQ(req[FQN_enum::kOut], kWriteTo);

    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4, DType> data;
    Tensor<xpu, 4, DType> out;
    if (in_data[FQN_enum::kData].ndim() == 2) {
      Shape<4> dshape = Shape4(in_data[FQN_enum::kData].shape_[0],
                               in_data[FQN_enum::kData].shape_[1], 1, 1);
      data = in_data[FQN_enum::kData].get_with_shape<xpu, 4, DType>(dshape, s);
      out = out_data[FQN_enum::kOut].get_with_shape<xpu, 4, DType>(dshape, s);
    } else {
      data = in_data[FQN_enum::kData].get<xpu, 4, DType>(s);
      out = out_data[FQN_enum::kOut].get<xpu, 4, DType>(s);
    }
    Tensor<xpu, 1, DType> mins = aux_states[FQN_enum::kMin].get<xpu, 1, DType>(s);
    Tensor<xpu, 1, DType> maxs = aux_states[FQN_enum::kMax].get<xpu, 1, DType>(s);

    if (param_.is_perchannel) {
      if (ctx.is_train > 0) {
        mins = minall_except_dim<0>(data);
        maxs = maxall_except_dim<0>(data);
        const ScalarExp<DType> eps(DType(1e-6));
        maxs = maxs + eps;
      }
      const ScalarExp<DType> quant_level_rev(1.0/QUANT_LEVEL);
      Assign(out, req[FQN_enum::kOut], 
             F<mshadow_op::round>((data - mshadow::expr::broadcast<0>(mins, data.shape_)) / \
                                   mshadow::expr::broadcast<0>((maxs - mins) * quant_level_rev, data.shape_)) * \
                                   mshadow::expr::broadcast<0>((maxs - mins) * quant_level_rev, data.shape_) + \
                                   mshadow::expr::broadcast<0>(mins, data.shape_));
    }
    else {
      if (ctx.is_train > 0) {
        mxnet::TShape src_shape, dst_shape;
        size_t temp_reduce_size;
        temp_reduce_size = ConfigReduce<xpu, DType>(
            s, data.shape_, mxnet::TShape(1, 1), &src_shape, &dst_shape);
          // space for temp_reudce_size, reduce_value
        Tensor<xpu, 1, uint8_t> workspace = ctx.requested[FQN_enum::kTempSpace]
            .get_space_typed<xpu, 1, uint8_t>(Shape1(temp_reduce_size + sizeof(DType)), s);
        
        uint64_t allocated_bytes = 0ULL;
        Tensor<xpu, 1, char> temp_reduce_space(reinterpret_cast<char*>(workspace.dptr_ + allocated_bytes), 
                                          Shape1(temp_reduce_size), s);
        allocated_bytes += temp_reduce_size;

        const int dev_id = ctx.run_ctx.ctx.dev_id;
        TBlob in_reduce_t(reinterpret_cast<DType *>(workspace.dptr_ + allocated_bytes), Shape1(1), xpu::kDevMask,
                      dev_id);
        Tensor<xpu, 1, DType> reduce_val = in_reduce_t.get<xpu, 1, DType>(s);
        allocated_bytes += sizeof(DType);
        
        // min value
        broadcast::Reduce<red::minimum, 2, DType, mshadow::op::identity>(
            s, in_reduce_t.reshape(dst_shape), kWriteTo, temp_reduce_space, in_data[0].reshape(src_shape));
        mshadow::Copy(mins, reduce_val, s);
        // max value
        broadcast::Reduce<red::maximum, 2, DType, mshadow::op::identity>(
            s, in_reduce_t.reshape(dst_shape), kWriteTo, temp_reduce_space, in_data[0].reshape(src_shape));
        mshadow::Copy(maxs, reduce_val, s);
      }
      const ScalarExp<DType> quant_level_rev(1.0/QUANT_LEVEL);
      Assign(out, req[FQN_enum::kOut], F<mshadow_op::round>((data - broadcast_scalar(mins, data.shape_)) / \
                                                             broadcast_scalar((maxs - mins) * quant_level_rev, data.shape_)) * \
                                                             broadcast_scalar((maxs - mins) * quant_level_rev, data.shape_) + \
                                                             broadcast_scalar(mins, data.shape_));
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
    if (out_grad[FQN_enum::kOut].ndim() == 2) {
      Shape<4> dshape = Shape4(out_grad[FQN_enum::kOut].shape_[0],
                               out_grad[FQN_enum::kOut].shape_[1], 1, 1);
      data = in_data[FQN_enum::kData].get_with_shape<xpu, 4, DType>(dshape, s);
      out = out_data[FQN_enum::kOut].get_with_shape<xpu, 4, DType>(dshape, s);

      out_data_grad = out_grad[FQN_enum::kOut].get_with_shape<xpu, 4, DType>(dshape, s);
      data_grad = in_grad[FQN_enum::kData].get_with_shape<xpu, 4, DType>(dshape, s);
    } else {
      data = in_data[FQN_enum::kData].get<xpu, 4, DType>(s);
      out = out_data[FQN_enum::kOut].get<xpu, 4, DType>(s);

      out_data_grad = out_grad[FQN_enum::kOut].get<xpu, 4, DType>(s);
      data_grad = in_grad[FQN_enum::kData].get<xpu, 4, DType>(s);
    }
    mshadow::Copy(data_grad, out_data_grad, s);
  }

 private:
  FQNPara param_;
  int QUANT_LEVEL;

};  // class FQNOp

template<typename xpu>
Operator* CreateOp(FQNPara type, int dtype);

#if DMLC_USE_CXX11
class FQNProp : public OperatorProperty {
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
    const TShape &dshape = in_shape->at(FQN_enum::kData);
    out_shape->clear();
    out_shape->push_back(dshape);
    
    Shape<1>  tmp_aux_shape = Shape1(1);
    if (param_.is_perchannel) {
      tmp_aux_shape = Shape1(dshape[0]);
    }

    aux_shape->clear();
    aux_shape->push_back(TShape(tmp_aux_shape));
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
    int n_aux = this->ListAuxiliaryStates().size();
    aux_type->clear();
    for (int i = 0; i < n_aux; ++i ) aux_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new FQNProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "_contrib_FQN";
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[FQN_enum::kOut], 
            in_data[FQN_enum::kData], 
            out_data[FQN_enum::kOut]};
  }

  std::vector<std::string> ListArguments() const override {
    return {"data"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output"};
  }

  std::vector<std::string> ListAuxiliaryStates() const override {
    return {"min", "max"};
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
  FQNPara param_;
};
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_FQN_INL_H_

