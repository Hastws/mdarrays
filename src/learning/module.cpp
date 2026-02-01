#include "learning/module.h"

#include <cmath>
#include <cstdlib>
#include <random>
#include <utility>

#include "exp/function.h"
#include "learning/init.h"

namespace Autoalg {
namespace Learning {
Linear::Linear(Index in_features, Index out_features)
    : weight_(Shape{out_features, in_features}, true),
      bias_(Shape{1, out_features}, true) {
  KaimingInitializer weight_init(weight_);
  weight_init.Init();
  UniformInitializer bias_init(bias_);
  bias_init.Init();
}

Mdarray Linear::Forward(const Mdarray &x) {
  Mdarray y1 = Operator::CreateOperationMatrixMul(
      x, Operator::CreateOperationMatrixTranspose(weight_));
  Mdarray y2 = y1 + bias_;
  return y2;
}

ParamsDict Linear::Parameters() {
  return {{"weight", weight_}, {"bias", bias_}};
}

LinearWithReLU::LinearWithReLU(Index in_features, Index out_features)
    : Linear(in_features, out_features) {}

Mdarray LinearWithReLU::Forward(const Mdarray &x) {
  Mdarray y1 = Operator::CreateOperationMatrixMul(
      x, Operator::CreateOperationMatrixTranspose(weight_));
  Mdarray y2 = Operator::CreateOperationRelu(y1 + bias_);
  return y2;
}
}  // namespace Learning
}  // namespace Autoalg

namespace Autoalg {
namespace Learning {
Conv2d::Conv2d(Index in_channels, Index out_channels, MatrixSize kernel_size,
               MatrixSize stride, MatrixSize padding)
    : in_channels_(in_channels),
      out_channels_(out_channels),
      kernel_size_(std::move(kernel_size)),
      stride_(std::move(stride)),
      padding_(std::move(padding)),
      weight_(Shape{out_channels_,
                    in_channels_ * kernel_size_.first * kernel_size_.second},
              true) {
  KaimingInitializer weight_init(weight_);
  weight_init.Init();
}

Mdarray Conv2d::Forward(const Mdarray &x) {
  auto col_exp =
      Operator::CreateOperationImgToCol(x, kernel_size_, stride_, padding_);

  Mdarray y1 = Operator::CreateOperationMatrixMul(
      Mdarray(col_exp), Operator::CreateOperationMatrixTranspose(weight_));

  auto &&conv_feat_size = col_exp.Impl().ConvFeatSize();
  Mdarray y2 = y1.View(
      {conv_feat_size.first, conv_feat_size.second, x.Size(0), out_channels_});
  Mdarray y3 = y2.Permute({2, 3, 0, 1});
  return y3;
}

ParamsDict Conv2d::Parameters() { return {{"weight", weight_}}; }

Conv2dWithReLU::Conv2dWithReLU(Index in_channels, Index out_channels,
                               const MatrixSize &kernel_size,
                               const MatrixSize &stride,
                               const MatrixSize &padding)
    : Conv2d(in_channels, out_channels, kernel_size, stride, padding) {}

Mdarray Conv2dWithReLU::Forward(const Mdarray &x) {
  auto col_exp =
      Operator::CreateOperationImgToCol(x, kernel_size_, stride_, padding_);

  Mdarray y1 = Operator::CreateOperationRelu(Operator::CreateOperationMatrixMul(
      Mdarray(col_exp), Operator::CreateOperationMatrixTranspose(weight_)));

  auto &conv_feat_size = col_exp.Impl().ConvFeatSize();
  Mdarray y2 = y1.View(
      {conv_feat_size.first, conv_feat_size.second, x.Size(0), out_channels_});
  Mdarray y3 = y2.Permute({2, 3, 0, 1});
  return y3;
}
}  // namespace Learning
}  // namespace Autoalg

namespace Autoalg {
namespace Learning {

MaxPool2d::MaxPool2d(MatrixSize kernel_size, MatrixSize stride,
                     MatrixSize padding)
    : kernel_size_(std::move(kernel_size)),
      stride_(std::move(stride)),
      padding_(std::move(padding)) {}

Mdarray MaxPool2d::Forward(const Mdarray &x) {
  auto col_exp =
      Operator::CreateOperationImgToCol(x, kernel_size_, stride_, padding_);
  Mdarray y1 = col_exp;

  auto &conv_feat_size = col_exp.Impl().ConvFeatSize();
  Mdarray y2 = y1.View({conv_feat_size.first, conv_feat_size.second, x.Size(0),
                        x.Size(1), kernel_size_.first * kernel_size_.second});
  Mdarray y3 = Operator::CreateOperationMax(y2, 4);
  Mdarray y4 = y3.Permute({2, 3, 0, 1});
  return y4;
}

ParamsDict MaxPool2d::Parameters() { return {}; }

Mdarray CrossEntropy::Forward(const Mdarray &input, const Index *labels) {
  auto log_its = Operator::CreateOperationLogSoftmax(input);
  auto nll = Operator::CreateOperationNllLoss(log_its, labels);
  Mdarray loss = Operator::CreateOperationMean(nll, 0);
  return loss;
}

// LayerNorm 实现
LayerNorm::LayerNorm(Index normalized_shape, BasicData eps)
    : normalized_shape_(normalized_shape),
      eps_(eps),
      gamma_(Shape{1, normalized_shape}, true),
      beta_(Shape{1, normalized_shape}, true) {
  // 手动初始化 gamma 为 1，beta 为 0
  StorageUniversalAgent gamma_agent(const_cast<MdarrayImpl &>(gamma_.Impl()).GetStorage());
  StorageUniversalAgent beta_agent(const_cast<MdarrayImpl &>(beta_.Impl()).GetStorage());
  BasicData *gamma_data = gamma_agent.GetStorageData();
  BasicData *beta_data = beta_agent.GetStorageData();
  for (Index i = 0; i < normalized_shape; ++i) {
    gamma_data[i] = 1.0;
    beta_data[i] = 0.0;
  }
}

Mdarray LayerNorm::Forward(const Mdarray &input) {
  // input: (..., normalized_shape)
  // 计算最后一维的均值和方差
  Index dims = input.DimensionsSize();
  Index last_dim = input.Size(dims - 1);
  
  // 将输入展平为 (N, C) 形状
  Index N = 1;
  for (Index i = 0; i < dims - 1; ++i) {
    N *= input.Size(i);
  }
  
  Mdarray x2d = input.View({N, last_dim});
  
  // 计算均值: mean over last dim
  Mdarray mean = Operator::CreateOperationMean(x2d, 1);  // (N, 1) -> (N,)
  
  // 扩展 mean 以便广播
  Mdarray mean_expanded = mean.Unsqueeze(1);  // (N, 1)
  
  // 计算 x - mean
  Mdarray x_centered = x2d - mean_expanded;  // (N, C)
  
  // 计算方差: var = mean((x - mean)^2)
  Mdarray x_sq = x_centered * x_centered;
  Mdarray var = Operator::CreateOperationMean(x_sq, 1);  // (N,)
  Mdarray var_expanded = var.Unsqueeze(1);  // (N, 1)
  
  // 添加 eps 并计算标准差的倒数
  IndexArray var_size = {var_expanded.Size(0), var_expanded.Size(1)};
  Mdarray eps_tensor = Operator::CreateOperationConstant(eps_, var_size);
  Mdarray var_plus_eps = var_expanded + eps_tensor;
  Mdarray std_inv = Operator::CreateOperationRsqrt(var_plus_eps);  // 1 / sqrt(var + eps)
  
  // 归一化
  Mdarray x_norm = x_centered * std_inv;  // (N, C)
  
  // 应用 gamma 和 beta (可学习参数)
  Mdarray y = x_norm * gamma_ + beta_;  // (N, C) * (1, C) + (1, C) -> (N, C)
  
  // 恢复原始形状
  return y.View(input.Size());
}

ParamsDict LayerNorm::Parameters() {
  return {{"gamma", gamma_}, {"beta", beta_}};
}

// Dropout 实现
Dropout::Dropout(BasicData p) : p_(p), training_(true) {}

Mdarray Dropout::Forward(const Mdarray &input) {
  if (!training_ || p_ == 0.0) {
    return input;
  }
  
  // 生成 dropout mask
  Index total_size = input.Size().SpaceSize();
  Shape shape = input.Size();
  
  // 创建随机 mask
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<BasicData> dis(0.0, 1.0);
  
  std::vector<BasicData> mask_data(total_size);
  BasicData scale = 1.0 / (1.0 - p_);  // 缩放因子，保持期望不变
  
  for (Index i = 0; i < total_size; ++i) {
    mask_data[i] = (dis(gen) > p_) ? scale : 0.0;
  }
  
  Mdarray mask(mask_data.data(), shape, false);
  
  return input * mask;
}

}  // namespace Learning
}  // namespace Autoalg