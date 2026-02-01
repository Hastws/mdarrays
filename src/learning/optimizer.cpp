#include <cmath>
#include <cstring>

#include "learning/optimizer.h"
#include "mdarray/mdarray.h"
#include "mdarray/mdarray_impl.h"
#include "mdarray/storage.h"

namespace Autoalg {
namespace Learning {

OptimizerBase::OptimizerBase(const ParamsDict &params_dict) {
  params_.reserve(params_dict.size());
  for (const auto &named_param_ref : params_dict) {
    Mdarray &ma = named_param_ref.second.get();
    auto &impl = const_cast<MdarrayImpl &>(ma.Impl());
    CHECK_TRUE(impl.IsContiguous(),
               "Only contiguous Mdarray can be optimized.")
    params_.emplace_back(impl);
  }
}

void OptimizerBase::ZeroGrad() {
  for (MdarrayImpl &t : params_) {
    BasicData *grad_data_ptr = GetGradData(t);
    std::memset(grad_data_ptr, 0, t.Size().SpaceSize() * sizeof(BasicData));
  }
}

StochasticGradientDescent::StochasticGradientDescent(
    const ParamsDict &params_dict, BasicData lr)
    : OptimizerBase(params_dict), lr_(lr) {}

void StochasticGradientDescent::Step() {
  for (MdarrayImpl &t : params_) {
    BasicData *storage_data_ptr = GetStorageData(t);
    BasicData *grad_data_ptr = GetGradData(t);
    Index data_size = GetDataSize(t);

    for (Index i = 0; i < data_size; ++i) {
      storage_data_ptr[i] -= lr_ * grad_data_ptr[i];
    }
  }
}

StochasticGradientDescentWithMomentum::StochasticGradientDescentWithMomentum(
    const ParamsDict &params_dict, BasicData lr, BasicData momentum)
    : OptimizerBase(params_dict),
      lr_(lr),
      momentum_(momentum),
      first_step_(true) {
  running_means_.reserve(params_.size());
  for (MdarrayImpl &t : params_) {
    Index n_bytes = sizeof(BasicData) * GetDataSize(t);
    running_means_.emplace_back(Allocator::UniqueAllocate<BasicData>(n_bytes));
  }
}

void StochasticGradientDescentWithMomentum::Step() {
  if (first_step_) {
    first_step_ = false;
    for (Index i = 0; i < params_.size(); ++i) {
      MdarrayImpl &t = params_[i];
      BasicData *storage_data_ptr = GetStorageData(t);
      BasicData *grad_data_ptr = GetGradData(t);
      BasicData *vx = running_means_[i].get();
      Index data_size = GetDataSize(t);

      std::memcpy(vx, grad_data_ptr, data_size * sizeof(BasicData));
      for (Index j = 0; j < data_size; ++j) {
        storage_data_ptr[j] -= lr_ * vx[j];
      }
    }
  } else {
    for (Index i = 0; i < params_.size(); ++i) {
      MdarrayImpl &t = params_[i];
      BasicData *storage_data_ptr = GetStorageData(t);
      BasicData *grad_data_ptr = GetGradData(t);
      BasicData *vx = running_means_[i].get();
      Index data_size = GetDataSize(t);

      for (Index j = 0; j < data_size; ++j) {
        vx[j] = momentum_ * vx[j] + grad_data_ptr[j];
        storage_data_ptr[j] -= lr_ * vx[j];
      }
    }
  }
}

// Adam 优化器实现
Adam::Adam(const ParamsDict &params_dict, BasicData lr, BasicData beta1,
           BasicData beta2, BasicData eps)
    : OptimizerBase(params_dict),
      lr_(lr),
      beta1_(beta1),
      beta2_(beta2),
      eps_(eps),
      t_(0) {
  m_.reserve(params_.size());
  v_.reserve(params_.size());
  
  for (MdarrayImpl &t : params_) {
    Index n_bytes = sizeof(BasicData) * GetDataSize(t);
    m_.emplace_back(Allocator::UniqueAllocate<BasicData>(n_bytes));
    v_.emplace_back(Allocator::UniqueAllocate<BasicData>(n_bytes));
    
    // 初始化为 0
    std::memset(m_.back().get(), 0, n_bytes);
    std::memset(v_.back().get(), 0, n_bytes);
  }
}

void Adam::Step() {
  ++t_;
  
  // 偏置修正系数
  BasicData bias_correction1 = 1.0 - std::pow(beta1_, t_);
  BasicData bias_correction2 = 1.0 - std::pow(beta2_, t_);
  
  for (Index i = 0; i < params_.size(); ++i) {
    MdarrayImpl &param = params_[i];
    BasicData *storage_data_ptr = GetStorageData(param);
    BasicData *grad_data_ptr = GetGradData(param);
    BasicData *m_ptr = m_[i].get();
    BasicData *v_ptr = v_[i].get();
    Index data_size = GetDataSize(param);
    
    for (Index j = 0; j < data_size; ++j) {
      BasicData g = grad_data_ptr[j];
      
      // 更新一阶矩估计
      m_ptr[j] = beta1_ * m_ptr[j] + (1.0 - beta1_) * g;
      
      // 更新二阶矩估计
      v_ptr[j] = beta2_ * v_ptr[j] + (1.0 - beta2_) * g * g;
      
      // 偏置修正
      BasicData m_hat = m_ptr[j] / bias_correction1;
      BasicData v_hat = v_ptr[j] / bias_correction2;
      
      // 更新参数
      storage_data_ptr[j] -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
    }
  }
}

// Warmup + 线性衰减调度器实现
WarmupLinearScheduler::WarmupLinearScheduler(BasicData base_lr, 
                                             Index warmup_steps,
                                             Index total_steps, 
                                             BasicData min_lr)
    : base_lr_(base_lr),
      warmup_steps_(warmup_steps),
      total_steps_(total_steps),
      min_lr_(min_lr) {}

BasicData WarmupLinearScheduler::GetLr(Index step) {
  if (step < warmup_steps_) {
    // Warmup 阶段: 从 0 线性增加到 base_lr
    return base_lr_ * static_cast<BasicData>(step + 1) / warmup_steps_;
  } else {
    // 线性衰减阶段
    Index decay_steps = total_steps_ - warmup_steps_;
    Index current_decay_step = step - warmup_steps_;
    BasicData decay_ratio = static_cast<BasicData>(decay_steps - current_decay_step) / decay_steps;
    return std::max(min_lr_, min_lr_ + (base_lr_ - min_lr_) * decay_ratio);
  }
}

// Warmup + 余弦退火调度器实现
WarmupCosineScheduler::WarmupCosineScheduler(BasicData base_lr,
                                             Index warmup_steps,
                                             Index total_steps,
                                             BasicData min_lr)
    : base_lr_(base_lr),
      warmup_steps_(warmup_steps),
      total_steps_(total_steps),
      min_lr_(min_lr) {}

BasicData WarmupCosineScheduler::GetLr(Index step) {
  if (step < warmup_steps_) {
    // Warmup 阶段: 从 0 线性增加到 base_lr
    return base_lr_ * static_cast<BasicData>(step + 1) / warmup_steps_;
  } else {
    // 余弦退火阶段
    Index decay_steps = total_steps_ - warmup_steps_;
    Index current_decay_step = step - warmup_steps_;
    BasicData progress = static_cast<BasicData>(current_decay_step) / decay_steps;
    BasicData cosine_decay = 0.5 * (1.0 + std::cos(M_PI * progress));
    return min_lr_ + (base_lr_ - min_lr_) * cosine_decay;
  }
}

}  // namespace Learning
}  // namespace Autoalg