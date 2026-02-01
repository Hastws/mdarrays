#ifndef MULTIDIMENSIONAL_ARRAYS_LEARNING_OPTIMIZER_H
#define MULTIDIMENSIONAL_ARRAYS_LEARNING_OPTIMIZER_H

#include <functional>
#include <vector>

#include "learning/module.h"

namespace Autoalg {
namespace Learning {
class OptimizerBase {
 public:
  explicit OptimizerBase(const ParamsDict &params_dict);
  void ZeroGrad();
  virtual void Step() = 0;

 protected:
  static Index GetDataSize(const MdarrayImpl &mdarray_impl) {
    return mdarray_impl.Size().SpaceSize();
  }
  static BasicData *GetStorageData(MdarrayImpl &mdarray_impl) {
    StorageUniversalAgent storage_universal_agent(mdarray_impl.GetStorage());
    return storage_universal_agent.GetStorageData();
  };
  static BasicData *GetGradData(MdarrayImpl &mdarray_impl) {
    MdarrayImplUniversalAgent mdarray_impl_universal_agent(mdarray_impl);
    auto &storage = mdarray_impl_universal_agent.GetGradMetaPtr()->grad_;
    StorageUniversalAgent storage_universal_agent(storage);
    return storage_universal_agent.GetStorageData();
  }

  std::vector<std::reference_wrapper<MdarrayImpl>> params_;
};

class StochasticGradientDescent : public OptimizerBase {
 public:
  StochasticGradientDescent(const ParamsDict &params_dict, BasicData lr);
  void Step() override;
  BasicData Lr() const { return lr_; }
  void SetLr(BasicData lr) { lr_ = lr; }

 private:
  BasicData lr_;
};

class StochasticGradientDescentWithMomentum : public OptimizerBase {
 public:
  StochasticGradientDescentWithMomentum(const ParamsDict &params_dict,
                                        BasicData lr, BasicData momentum);
  void Step() override;
  BasicData Lr() const { return lr_; }
  void SetLr(BasicData lr) { lr_ = lr; }
  void LrDecay(BasicData factor) { lr_ *= factor; }

 private:
  BasicData lr_;
  BasicData momentum_;
  bool first_step_;
  std::vector<Allocator::UniquePtr<BasicData>> running_means_;
};

// Adam 优化器
class Adam : public OptimizerBase {
 public:
  Adam(const ParamsDict &params_dict, BasicData lr = 0.001,
       BasicData beta1 = 0.9, BasicData beta2 = 0.999, BasicData eps = 1e-8);
  
  void Step() override;
  
  BasicData Lr() const { return lr_; }
  void SetLr(BasicData lr) { lr_ = lr; }

 private:
  BasicData lr_;
  BasicData beta1_;
  BasicData beta2_;
  BasicData eps_;
  Index t_;  // timestep
  std::vector<Allocator::UniquePtr<BasicData>> m_;  // first moment
  std::vector<Allocator::UniquePtr<BasicData>> v_;  // second moment
};

// 学习率调度器基类
class LrSchedulerBase {
 public:
  virtual ~LrSchedulerBase() = default;
  virtual BasicData GetLr(Index step) = 0;
};

// Warmup + 线性衰减调度器
class WarmupLinearScheduler : public LrSchedulerBase {
 public:
  WarmupLinearScheduler(BasicData base_lr, Index warmup_steps, 
                        Index total_steps, BasicData min_lr = 0.0);
  
  BasicData GetLr(Index step) override;

 private:
  BasicData base_lr_;
  Index warmup_steps_;
  Index total_steps_;
  BasicData min_lr_;
};

// Warmup + 余弦退火调度器
class WarmupCosineScheduler : public LrSchedulerBase {
 public:
  WarmupCosineScheduler(BasicData base_lr, Index warmup_steps, 
                        Index total_steps, BasicData min_lr = 0.0);
  
  BasicData GetLr(Index step) override;

 private:
  BasicData base_lr_;
  Index warmup_steps_;
  Index total_steps_;
  BasicData min_lr_;
};

}  // namespace Learning
}  // namespace Autoalg
#endif