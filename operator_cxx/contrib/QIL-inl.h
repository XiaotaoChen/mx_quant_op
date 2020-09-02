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
 * Copyright (c) 2015 by Contributors
 * \file QIL-inl.h
 * \brief
 * \author Xiaotao Chen
*/
#ifndef MXNET_OPERATOR_QIL_INL_H_
#define MXNET_OPERATOR_QIL_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include <cmath>
#include "../operator_common.h"
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "../tensor/control_flow_op.h"
#include "../quantization/quantization_utils.h"

namespace mxnet {
namespace op {

namespace QIL_enum {
enum QILOpInputs {kData, kCenter, kDistance, kGamma};
enum QILOpOutputs {kOut};
enum QILOpResource {kTempSpace};
}  // namespace QIL_enum

struct QILPara : public dmlc::Parameter<QILPara> {
  bool is_weight;
  bool fix_gamma;
  int nbits;
  DMLC_DECLARE_PARAMETER(QILPara) {
    DMLC_DECLARE_FIELD(is_weight).set_default(false)
    .describe("the quantization is for weight or not. defalut is False");
    DMLC_DECLARE_FIELD(fix_gamma).set_default(true)
    .describe("fix the gamma of for weight transform or not. defalut is True");
    DMLC_DECLARE_FIELD(nbits).set_default(8)
    .describe("the quantized target bits. defalut is 8");
    
  }
};

template <typename DType, typename xpu>
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

// currently don't consider the gamma, the default is to 1.
template<int req>
struct transformer {
    template<typename DType>
    MSHADOW_XINLINE static void Map(index_t i, DType* out, const DType* data, 
                                    const real_t* center, const real_t* distance) {
        DType low_bound = DType(*center - *distance);
        DType high_bound = DType(*center + *distance);
        int sign = 1;
        DType tmp = data[i];
        if (data[i] < 0) {
            sign = -sign;
            tmp = -tmp;
        }
        if (tmp < low_bound) {
            KERNEL_ASSIGN(out[i], req, DType(0.0f));
            // out[i] = DType(0.0f);
        }
        else if (tmp > high_bound) {
            KERNEL_ASSIGN(out[i], req, DType(sign));
            // out[i] = DType(sign);
        }
        else {
            // out = [ 0.5*(|data| - center)/distance + 0.5 ] * sign
            KERNEL_ASSIGN(out[i], req, DType(sign) * (DType(0.5f) * (tmp - *center) / *distance + DType(0.5f)));
            // out[i] = DType(sign) * (DType(0.5f) * (tmp - *center) / *distance + DType(0.5f));
        }
    }
};

template<int req>
struct clip_gradient {
    template<typename DType>
    MSHADOW_XINLINE static void Map(index_t i, DType* out, const DType* data, 
                                    const real_t* center, const real_t* distance) {
        DType low_bound = DType(*center - *distance);
        DType high_bound = DType(*center + *distance);
        DType tmp;
        if (data[i] < 0) {
            tmp = -data[i];
        }
        else {
            tmp = data[i];
        }
        if (tmp < low_bound || tmp > high_bound) {
            out[i] = DType(0.0f);
        }
    }
};

template<int req>
struct fetch_sign {
    template<typename DType>
    MSHADOW_XINLINE static void Map(index_t i, DType* out, const DType* data) {
        if (data[i] >= 0) {
            out[i] = DType(1);
        }
        else {
            out[i] = DType(-1);
        }
    }
};

// the max abs store in max_ptr
struct find_maxabs {
  MSHADOW_XINLINE static void Map(int i, real_t *imin_range, real_t* imax_range) {
    if (i < 1){
      *imax_range = MaxAbs(*imin_range, *imax_range);
    }
  }
};

template<typename xpu>
void find_max(const OpContext &ctx, const TBlob &data, mshadow::Stream<xpu> *s, 
              mshadow::Tensor<xpu, 1, char> &temp_reduce_space, TBlob &in_min_t, TBlob &in_max_t,
              const mxnet::TShape &src_shape, const mxnet::TShape &dst_shape){
    using namespace mshadow;
    using namespace mshadow::expr;
    broadcast::Reduce<red::minimum, 2, real_t, mshadow::op::identity>(
        s, in_min_t.reshape(dst_shape), kWriteTo, temp_reduce_space, data.reshape(src_shape));
    broadcast::Reduce<red::maximum, 2, real_t, mshadow::op::identity>(
        s, in_max_t.reshape(dst_shape), kWriteTo, temp_reduce_space, data.reshape(src_shape));

    // the maxabs value is save in in_max_t
    mxnet_op::Kernel<find_maxabs, xpu>::Launch(s, 1, in_min_t.dptr<real_t>(), in_max_t.dptr<real_t>());
}

template<typename xpu, typename DType>
class QILOp : public Operator {
 public:
  explicit QILOp(QILPara param) {
    this->param_ = param;
    QUANT_LEVEL = std::pow(2, param.nbits) - 1;
    init = true;
  }


  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    using namespace mxnet_op;
    CHECK_EQ(in_data.size(), 4U);
    CHECK_EQ(out_data.size(), 1U);
    CHECK_EQ(req.size(), 1U);
    CHECK_EQ(req[QIL_enum::kOut], kWriteTo);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Stream<cpu> *s_cpu = ctx.get_stream<cpu>();
    Tensor<xpu, 4, DType> data;
    Tensor<xpu, 4, DType> out;
    Tensor<xpu, 1, real_t> center;
    Tensor<xpu, 1, real_t> distance;
    Tensor<xpu, 1, real_t> gamma;

    if (in_data[QIL_enum::kData].ndim() == 2) {
      Shape<4> dshape = Shape4(in_data[QIL_enum::kData].shape_[0],
                               in_data[QIL_enum::kData].shape_[1], 1, 1);
      data = in_data[QIL_enum::kData].get_with_shape<xpu, 4, DType>(dshape, s);
      out = out_data[QIL_enum::kOut].get_with_shape<xpu, 4, DType>(dshape, s);
    } else {
      data = in_data[QIL_enum::kData].get<xpu, 4, DType>(s);
      out = out_data[QIL_enum::kOut].get<xpu, 4, DType>(s);
    }
    center = in_data[QIL_enum::kCenter].get<xpu, 1, real_t>(s);
    distance = in_data[QIL_enum::kDistance].get<xpu, 1, real_t>(s);
    gamma = in_data[QIL_enum::kGamma].get<xpu, 1, real_t>(s);

    CHECK_EQ(param_.fix_gamma, true) << "currently only support fix gamma mode.";
    // initialize distance =max(abs(data)), center=0
    if (init) {
        uint64_t WORKSPACE_LIMIT = 1024; // allocate 1k
        Tensor<xpu, 1, uint8_t> workspace = ctx.requested[QIL_enum::kTempSpace]
          .get_space_typed<xpu, 1, uint8_t>(Shape1(WORKSPACE_LIMIT), s);
        uint64_t allocated_bytes = 0ULL;

        mxnet::TShape src_shape, dst_shape;
        const size_t temp_reduce_size = ConfigReduce<xpu, real_t>(
            s, data.shape_, mxnet::TShape(1, 1), &src_shape, &dst_shape);
        Tensor<xpu, 1, char> temp_reduce_space(reinterpret_cast<char*>(workspace.dptr_ + allocated_bytes), 
                                       Shape1(temp_reduce_size), s);
        allocated_bytes += temp_reduce_size;
        const int dev_id = ctx.run_ctx.ctx.dev_id;
        TBlob in_min_t(reinterpret_cast<real_t *>(workspace.dptr_ + allocated_bytes), Shape1(1), xpu::kDevMask,
                      dev_id);
        allocated_bytes += sizeof(real_t);
        TBlob in_max_t(reinterpret_cast<real_t *>(workspace.dptr_ + allocated_bytes), Shape1(1), xpu::kDevMask,
                      dev_id);
        allocated_bytes += sizeof(real_t);

        Tensor<xpu, 1, real_t> max_val = in_max_t.get<xpu, 1, real_t>(s);
        find_max<xpu>(ctx, in_data[0], s, temp_reduce_space, in_min_t, in_max_t, src_shape, dst_shape);
        mshadow::Copy(distance, max_val, s);

        real_t tmp_center = 0.0f;
        Tensor<cpu, 1, real_t> tmp_center_tensor(&tmp_center, Shape1(1), s_cpu);
        mshadow::Copy(center, tmp_center_tensor, s);
        init = false;
    }

    // // for debug
    // real_t tmp_center = 0.0f;
    // real_t tmp_distance = 0.0f;
    // Tensor<cpu, 1, real_t> tmp_center_tensor(&tmp_center, Shape1(1), s_cpu);
    // Tensor<cpu, 1, real_t> tmp_distance_tensor(&tmp_distance, Shape1(1), s_cpu);
    // mshadow::Copy(tmp_center_tensor, center, s);
    // mshadow::Copy(tmp_distance_tensor, distance, s);

    // transformer
    Kernel<transformer<kWriteTo>, xpu>::Launch(s, data.shape_.Size(), out.dptr_, data.dptr_, center.dptr_, distance.dptr_);
    // quantizer quantize the transformed weight
    const ScalarExp<DType> quant_level(QUANT_LEVEL);
    Assign(out, req[QIL_enum::kOut], F<mshadow_op::round>(out * quant_level) / quant_level)
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
    CHECK_EQ(out_grad.size(), 1U);
    CHECK_EQ(in_data.size(), 4U);
    CHECK_EQ(out_data.size(), 1U);
    CHECK_EQ(in_grad.size(), 4U);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Stream<cpu> *s_cpu = ctx.get_stream<cpu>();
    Tensor<xpu, 4, DType> data_grad;
    Tensor<xpu, 4, DType> out_data_grad;
    Tensor<xpu, 4, DType> data;
    Tensor<xpu, 4, DType> out;

    Tensor<xpu, 1, real_t> center;
    Tensor<xpu, 1, real_t> distance;
    Tensor<xpu, 1, real_t> gamma;
    Tensor<xpu, 1, real_t> center_grad;
    Tensor<xpu, 1, real_t> distance_grad;
    Tensor<xpu, 1, real_t> gamma_grad;

    if (out_grad[QIL_enum::kOut].ndim() == 2) {
      Shape<4> dshape = Shape4(out_grad[QIL_enum::kOut].shape_[0],
                               out_grad[QIL_enum::kOut].shape_[1], 1, 1);
      data = in_data[QIL_enum::kData].get_with_shape<xpu, 4, DType>(dshape, s);
      out = out_data[QIL_enum::kOut].get_with_shape<xpu, 4, DType>(dshape, s);

      out_data_grad = out_grad[QIL_enum::kOut].get_with_shape<xpu, 4, DType>(dshape, s);
      data_grad = in_grad[QIL_enum::kData].get_with_shape<xpu, 4, DType>(dshape, s);
    } else {
      data = in_data[QIL_enum::kData].get<xpu, 4, DType>(s);
      out = out_data[QIL_enum::kOut].get<xpu, 4, DType>(s);

      out_data_grad = out_grad[QIL_enum::kOut].get<xpu, 4, DType>(s);
      data_grad = in_grad[QIL_enum::kData].get<xpu, 4, DType>(s);
    }


    center = in_data[QIL_enum::kCenter].get<xpu, 1, real_t>(s);
    distance = in_data[QIL_enum::kDistance].get<xpu, 1, real_t>(s);
    gamma = in_data[QIL_enum::kGamma].get<xpu, 1, real_t>(s);

    center_grad = in_grad[QIL_enum::kCenter].get<xpu, 1, real_t>(s);
    distance_grad = in_grad[QIL_enum::kDistance].get<xpu, 1, real_t>(s);
    gamma_grad = in_grad[QIL_enum::kGamma].get<xpu, 1, real_t>(s);

    // print_data<DType, xpu>(data, s, ctx, "data");
    // print_data<DType, xpu>(out_data_grad, s, ctx, "out data grad");
    // print_data<DType, xpu>(data_grad, s, ctx, "data grad");
    // real_t tmp = 0.0f;
    // Tensor<cpu, 1, real_t> tmp_tensor(&tmp, Shape1(1), s_cpu);
    // mshadow::Copy(tmp_tensor, center, s);
    // std::cout<<"center: "<< tmp<<std::endl;
    // mshadow::Copy(tmp_tensor, distance, s);
    // std::cout<<"distance: "<< tmp<<std::endl;

    Kernel<clip_gradient<kWriteTo>, xpu>::Launch(s, out_data_grad.shape_.Size(), out_data_grad.dptr_, data.dptr_, center.dptr_, distance.dptr_);
    // print_data<DType, xpu>(out_data_grad, s, ctx, "out data grad after clipped");


    // copy center and distance var to cpu mem
    real_t center_scalar = 0.0f;
    real_t distance_scalar = 0.0f;
    Tensor<cpu, 1, real_t> center_tensor(&center_scalar, Shape1(1), s_cpu);
    Tensor<cpu, 1, real_t> distance_tensor(&distance_scalar, Shape1(1), s_cpu);
    mshadow::Copy(center_tensor, center, s);
    mshadow::Copy(distance_tensor, distance, s);
    
    const ScalarExp<DType> tmp_distance(distance_scalar);
    const ScalarExp<DType> tmp_half(0.5f);
    const ScalarExp<DType> tmp_neg_half(-0.5f);

    // x = 0.5 * |x| / distance - 0.5 * center / distance + 0.5
    // the deri of data_i = 0.5 / distance
     Assign(data_grad, req[QIL_enum::kData], out_data_grad * tmp_half / tmp_distance);
     
    //  print_data<DType, xpu>(data_grad, s, ctx, "data grad after updated");

    // Compute necessary data for the reduce operation.
    mxnet::TShape src_shape, dst_shape;
    // to store sum result
    const size_t actual_float_size = sizeof(float);
    const size_t temp_reduce_size = ConfigReduce<xpu, real_t>(
        s, out_data_grad.shape_, mxnet::TShape(1, 1), &src_shape, &dst_shape);
    
    /*
    the 4 spaces: 1. sum_value
                  2. temp space for reduce
                  3. sign array of data
                  4. tmp_grad array for calculating the gradient of center and distance
    */
    Tensor<xpu, 1, char> temp_space = ctx.requested[QIL_enum::kTempSpace].get_space_typed<xpu, 1, char>(
        Shape1(data.shape_.Size() * sizeof(DType) + actual_float_size + 
               temp_reduce_size + out_data_grad.shape_.Size() * sizeof(DType)), s);
    uint64_t allocated_bytes = 0ULL;
    // sign array
    Tensor<xpu, 4, DType> sign_arr(reinterpret_cast<DType*>(temp_space.dptr_ + allocated_bytes), data.shape_, s);
    allocated_bytes += sign_arr.shape_.Size() * sizeof(DType);
    // tmp grad
    const int dev_id = ctx.run_ctx.ctx.dev_id;
    TBlob tmp_grad_blob(reinterpret_cast<DType *>(temp_space.dptr_ + allocated_bytes), 
                        out_data_grad.shape_, xpu::kDevMask, dev_id);
    Tensor<xpu, 4, DType> tmp_grad = tmp_grad_blob.get<xpu, 4, DType>(s);
    allocated_bytes += tmp_grad.shape_.Size() * sizeof(DType);
    // sum value
    TBlob sum_t(reinterpret_cast<real_t *>(temp_space.dptr_), Shape1(1), xpu::kDevMask,
                    dev_id);
    allocated_bytes += actual_float_size;
    // temp space for reduce
    Tensor<xpu, 1, char> workspace(temp_space.dptr_ + allocated_bytes,
                                    Shape1(temp_reduce_size), s);
    allocated_bytes += temp_reduce_size;


    Kernel<fetch_sign<kWriteTo>, xpu>::Launch(s, data.shape_.Size(), sign_arr.dptr_, data.dptr_);
    
    // print_data<DType, xpu>(sign_arr, s, ctx, "sign array");

    // deri_center = (-0.5 / distance) * sign(x)
    // the gradient of center = sum(out_data_grad * deri_center)
    Assign(tmp_grad, kWriteTo, out_data_grad * tmp_neg_half / tmp_distance * sign_arr);
    // sum(center gradient)
    broadcast::Reduce<mshadow_op::sum, 2, real_t, mshadow::op::identity>(
        s, sum_t.reshape(dst_shape), kWriteTo, workspace, tmp_grad_blob.reshape(src_shape));
    
    Tensor<xpu, 1, real_t> grad_sum = sum_t.get<xpu, 1, real_t>(s);
    // assign the center gradient to center_grad
    mshadow::Copy(center_grad, grad_sum, s);

    // // for debug
    // mshadow::Copy(tmp_tensor, grad_sum, s);
    // std::cout<<"center gradient sum: "<<tmp <<std::endl;

    /*
    the gradient of distance
    deri_distance = -0.5 * (x_i - center) / distance^2, if x_i >= 0,
                  = -0.5 * (x_i + center) / distance^2, if x_i < 0
                  = -0.5 * (x_i - center * sign(x_i)) / distance^2
    the gradient of distance = deri_distance * grad
    */
    const ScalarExp<DType> tmp_center(center_scalar);
    Assign(tmp_grad, kWriteTo, out_data_grad * tmp_neg_half * (data - tmp_center * sign_arr) / (tmp_distance * tmp_distance))

    broadcast::Reduce<mshadow_op::sum, 2, real_t, mshadow::op::identity>(
        s, sum_t.reshape(dst_shape), kWriteTo, workspace, tmp_grad_blob.reshape(src_shape));
    // assign the distance gradient to center_grad
    mshadow::Copy(distance_grad, grad_sum, s);

    // // for debug
    // mshadow::Copy(tmp_tensor, grad_sum, s);
    // std::cout<<"distance grad: "<< tmp <<std::endl;

  }

 private:
  QILPara param_;
  int QUANT_LEVEL;
  bool init;

};  // class QILOp

template<typename xpu>
Operator* CreateOp(QILPara type, int dtype);

#if DMLC_USE_CXX11
class QILProp : public OperatorProperty {
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

    CHECK_EQ(in_shape->size(), 4U) << "Input:[data, center, distance, gamma]";
    const TShape &dshape = in_shape->at(QIL_enum::kData);
    if (!mxnet::ndim_is_known(dshape)) return false;
    in_shape->at(QIL_enum::kCenter) = mxnet::TShape(Shape1(1));
    in_shape->at(QIL_enum::kDistance) = mxnet::TShape(Shape1(1));
    in_shape->at(QIL_enum::kGamma) = mxnet::TShape(Shape1(1));
    out_shape->clear();
    out_shape->push_back(dshape);
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    using namespace mshadow;
    CHECK_GE(in_type->size(), 1U);
    int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "First input must have specified type";
    int dtype_param = (dtype == kFloat16) ? kFloat32 : dtype;
    for (size_t i = 1; i < in_type->size(); ++i) {
      if ((*in_type)[i] == -1) {
        (*in_type)[i] = dtype_param;
      } else {
        UNIFORM_TYPE_CHECK((*in_type)[i], dtype, ListArguments()[i]);
      }
    }

    out_type->clear();
    out_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new QILProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "_contrib_QIL";
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[QIL_enum::kOut],
            in_data[QIL_enum::kData], 
            in_data[QIL_enum::kCenter],
            in_data[QIL_enum::kDistance],
            in_data[QIL_enum::kGamma],};
  }


  std::vector<std::string> ListArguments() const override {
    return {"data", "center", "distance", "gamma"};
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
  QILPara param_;
};
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_Qunatization_Int8_INL_H_

