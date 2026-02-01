/**
 * CIFAR-10 模型横向对比测试
 * 对比 MLP, CNN, Transformer 在 CIFAR-10 数据集上的表现
 */

#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "data/data.h"
#include "exp/function.h"
#include "learning/module.h"
#include "learning/optimizer.h"
#include "mdarray/mdarray.h"
#include "mdarray/mdarray_impl.h"
#include "mdarray/shape.h"
#include "mdarray/storage.h"
#include "utils/exception.h"

using namespace Autoalg;

//=============================================================================
// MLP 模型 - 简单全连接网络
//=============================================================================
class MLP_Cifar10 : public Learning::Module {
 public:
  MLP_Cifar10()
      : linear1_(3 * 32 * 32, 512),
        linear2_(512, 256),
        linear3_(256, 128),
        linear4_(128, 10) {}

  Mdarray Forward(const Mdarray &input) override {
    // input: (B, 3, 32, 32) -> flatten to (B, 3072)
    Index batch = input.Size(0);
    Mdarray flat = input.View({batch, 3 * 32 * 32});
    Mdarray h1 = linear1_.Forward(flat);
    Mdarray h2 = linear2_.Forward(h1);
    Mdarray h3 = linear3_.Forward(h2);
    Mdarray out = linear4_.Forward(h3);
    return out;
  }

  Learning::ParamsDict Parameters() override {
    return {{"linear1", linear1_.Parameters()},
            {"linear2", linear2_.Parameters()},
            {"linear3", linear3_.Parameters()},
            {"linear4", linear4_.Parameters()}};
  }

 private:
  Learning::LinearWithReLU linear1_;
  Learning::LinearWithReLU linear2_;
  Learning::LinearWithReLU linear3_;
  Learning::Linear linear4_;
};

//=============================================================================
// CNN 模型 - LeNet 风格卷积网络
//=============================================================================
class CNN_Cifar10 : public Learning::Module {
 public:
  CNN_Cifar10()
      : conv1_(3, 32, {3, 3}, {1, 1}, {1, 1}),   // 32x32 -> 32x32
        conv2_(32, 64, {3, 3}, {1, 1}, {1, 1}),  // 16x16 -> 16x16
        conv3_(64, 64, {3, 3}, {1, 1}, {1, 1}),  // 8x8 -> 8x8
        linear1_(64 * 4 * 4, 256),
        linear2_(256, 10) {}

  Mdarray Forward(const Mdarray &input) override {
    // Conv block 1
    Mdarray c1 = conv1_.Forward(input);      // (B, 32, 32, 32)
    Mdarray p1 = pool_.Forward(c1);          // (B, 32, 16, 16)

    // Conv block 2
    Mdarray c2 = conv2_.Forward(p1);         // (B, 64, 16, 16)
    Mdarray p2 = pool_.Forward(c2);          // (B, 64, 8, 8)

    // Conv block 3
    Mdarray c3 = conv3_.Forward(p2);         // (B, 64, 8, 8)
    Mdarray p3 = pool_.Forward(c3);          // (B, 64, 4, 4)

    // Flatten and FC
    Index batch = p3.Size(0);
    Mdarray feat(p3.Size());
    feat = p3;  // 触发计算
    Mdarray flat = feat.View({batch, 64 * 4 * 4});
    Mdarray h = linear1_.Forward(flat);
    Mdarray out = linear2_.Forward(h);
    return out;
  }

  Learning::ParamsDict Parameters() override {
    return {{"conv1", conv1_.Parameters()},
            {"conv2", conv2_.Parameters()},
            {"conv3", conv3_.Parameters()},
            {"linear1", linear1_.Parameters()},
            {"linear2", linear2_.Parameters()}};
  }

 private:
  Learning::Conv2dWithReLU conv1_;
  Learning::Conv2dWithReLU conv2_;
  Learning::Conv2dWithReLU conv3_;
  Learning::MaxPool2d pool_{{2, 2}, {2, 2}, {0, 0}};
  Learning::LinearWithReLU linear1_;
  Learning::Linear linear2_;
};

//=============================================================================
// Vision Transformer (ViT) - 简化版本
//=============================================================================

// Multi-Head Self-Attention
class MultiHeadAttention : public Learning::Module {
 public:
  MultiHeadAttention(Index d_model, Index n_heads)
      : d_model_(d_model),
        n_heads_(n_heads),
        d_head_(d_model / n_heads),
        q_proj_(d_model, d_model),
        k_proj_(d_model, d_model),
        v_proj_(d_model, d_model),
        o_proj_(d_model, d_model) {
    CHECK_TRUE(d_model_ % n_heads_ == 0, "d_model must be divisible by n_heads");
  }

  Mdarray Forward(const Mdarray &x_bsd) override {
    const Index B = x_bsd.Size(0);
    const Index S = x_bsd.Size(1);
    const Index D = x_bsd.Size(2);

    // Linear projections
    Mdarray x_2d = x_bsd.View({B * S, D});
    Mdarray q = q_proj_.Forward(x_2d).View({B, S, D});
    Mdarray k = k_proj_.Forward(x_2d).View({B, S, D});
    Mdarray v = v_proj_.Forward(x_2d).View({B, S, D});

    // Reshape to heads: (B, H, S, Dh)
    Mdarray q_bhsd = q.View({B, S, n_heads_, d_head_}).Permute({0, 2, 1, 3});
    Mdarray k_bhsd = k.View({B, S, n_heads_, d_head_}).Permute({0, 2, 1, 3});
    Mdarray v_bhsd = v.View({B, S, n_heads_, d_head_}).Permute({0, 2, 1, 3});

    // Merge (B,H) -> BH for batch matmul
    Mdarray q_bh = q_bhsd.Contiguous().View({B * n_heads_, S, d_head_});
    Mdarray k_bh = k_bhsd.Contiguous().View({B * n_heads_, S, d_head_});
    Mdarray v_bh = v_bhsd.Contiguous().View({B * n_heads_, S, d_head_});

    // Attention scores
    Mdarray kt = Operator::CreateOperationBatchMatrixTranspose(k_bh);
    Mdarray scores = Operator::CreateOperationBatchMatrixMul(q_bh, kt);

    // Scale
    const BasicData scale = 1.0 / std::sqrt(static_cast<double>(d_head_));
    Mdarray scale_const = Operator::CreateOperationConstant(
        scale, {scores.Size(0), scores.Size(1), scores.Size(2)});
    scores = scores * scale_const;

    // Softmax
    Mdarray attn = ApplySoftmax(scores);

    // Attention output
    Mdarray ctx = Operator::CreateOperationBatchMatrixMul(attn, v_bh);

    // Reshape back
    Mdarray ctx_bhsd = ctx.View({B, n_heads_, S, d_head_})
                           .Permute({0, 2, 1, 3}).Contiguous();
    Mdarray ctx_bsd = ctx_bhsd.View({B * S, d_model_});
    Mdarray out = o_proj_.Forward(ctx_bsd).View({B, S, d_model_});
    return out;
  }

  Learning::ParamsDict Parameters() override {
    return {{"q_proj", q_proj_.Parameters()},
            {"k_proj", k_proj_.Parameters()},
            {"v_proj", v_proj_.Parameters()},
            {"o_proj", o_proj_.Parameters()}};
  }

 private:
  // 对最后一维做 softmax
  Mdarray ApplySoftmax(const Mdarray &x) {
    // 先实体化
    Mdarray src(x.Size(), true);
    src = x;

    Index dims = src.DimensionsSize();
    Index C = src.Size(dims - 1);
    Index N = 1;
    for (Index i = 0; i < dims - 1; ++i) N *= src.Size(i);

    Mdarray x2d = src.View({N, C});
    Mdarray y2d(x2d.Size(), true);
    y2d = Operator::CreateOperationSoftmax(x2d);
    return y2d.View(src.Size());
  }

  Index d_model_, n_heads_, d_head_;
  Learning::Linear q_proj_, k_proj_, v_proj_, o_proj_;
};

// Feed Forward Network
class FeedForward : public Learning::Module {
 public:
  FeedForward(Index d_model, Index d_hidden)
      : fc1_(d_model, d_hidden), fc2_(d_hidden, d_model) {}

  Mdarray Forward(const Mdarray &x_bsd) override {
    Index B = x_bsd.Size(0), S = x_bsd.Size(1), D = x_bsd.Size(2);
    Mdarray x2d = x_bsd.View({B * S, D});
    Mdarray h = fc1_.Forward(x2d);
    Mdarray y2d = fc2_.Forward(h);
    return y2d.View({B, S, D});
  }

  Learning::ParamsDict Parameters() override {
    return {{"fc1", fc1_.Parameters()}, {"fc2", fc2_.Parameters()}};
  }

 private:
  Learning::LinearWithReLU fc1_;
  Learning::Linear fc2_;
};

// Transformer Encoder Block
class TransformerBlock : public Learning::Module {
 public:
  TransformerBlock(Index d_model, Index n_heads, Index d_ff)
      : attn_(d_model, n_heads), ffn_(d_model, d_ff) {}

  Mdarray Forward(const Mdarray &x_bsd) override {
    Mdarray a = attn_.Forward(x_bsd);
    Mdarray x1 = x_bsd + a;
    Mdarray f = ffn_.Forward(x1);
    Mdarray x2 = x1 + f;
    return x2;
  }

  Learning::ParamsDict Parameters() override {
    return {{"attn", attn_.Parameters()}, {"ffn", ffn_.Parameters()}};
  }

 private:
  MultiHeadAttention attn_;
  FeedForward ffn_;
};

// Vision Transformer for CIFAR-10
class ViT_Cifar10 : public Learning::Module {
 public:
  // patch_size: 将图像分成 patch_size x patch_size 的块
  // CIFAR-10: 32x32, patch_size=4 -> 8x8=64 patches
  ViT_Cifar10(Index patch_size = 4, Index d_model = 64, Index n_heads = 4,
              Index d_ff = 128, Index n_layers = 2, Index n_classes = 10)
      : patch_size_(patch_size),
        d_model_(d_model),
        n_heads_(n_heads),
        d_ff_(d_ff),
        n_layers_(n_layers),
        num_patches_((32 / patch_size) * (32 / patch_size)),
        patch_dim_(3 * patch_size * patch_size),
        patch_embed_(patch_dim_, d_model),
        head_(d_model, n_classes),
        pe_(Shape({1, (32/patch_size)*(32/patch_size), d_model})) {
    // 创建 Transformer blocks
    for (Index i = 0; i < n_layers_; ++i) {
      blocks_.emplace_back(new TransformerBlock(d_model, n_heads, d_ff));
    }

    // 初始化位置编码
    BuildSinusoidalPE();
  }

  Mdarray Forward(const Mdarray &input) override {
    // input: (B, 3, 32, 32)
    const Index B = input.Size(0);
    const Index C = input.Size(1);
    const Index H = input.Size(2);
    const Index W = input.Size(3);
    const Index P = patch_size_;
    const Index num_patches_h = H / P;
    const Index num_patches_w = W / P;

    // 将图像分成 patches: (B, num_patches, patch_dim)
    std::vector<BasicData> patches_data(B * num_patches_ * patch_dim_);
    
    // 先把 input 实体化
    Mdarray input_copy(input.Size());
    input_copy = input;
    
    // 获取数据指针
    StorageUniversalAgent agent(const_cast<MdarrayImpl &>(input_copy.Impl()).GetStorage());
    const BasicData* in_data = agent.GetStorageData();

    for (Index b = 0; b < B; ++b) {
      for (Index ph = 0; ph < num_patches_h; ++ph) {
        for (Index pw = 0; pw < num_patches_w; ++pw) {
          Index patch_idx = ph * num_patches_w + pw;
          for (Index c = 0; c < C; ++c) {
            for (Index i = 0; i < P; ++i) {
              for (Index j = 0; j < P; ++j) {
                Index img_h = ph * P + i;
                Index img_w = pw * P + j;
                Index img_offset = b * C * H * W + c * H * W + img_h * W + img_w;
                Index patch_offset = b * num_patches_ * patch_dim_ +
                                     patch_idx * patch_dim_ +
                                     c * P * P + i * P + j;
                patches_data[patch_offset] = in_data[img_offset];
              }
            }
          }
        }
      }
    }

    Mdarray patches(patches_data.data(), {B, num_patches_, patch_dim_}, false);

    // Patch embedding: (B*num_patches, patch_dim) -> (B*num_patches, d_model)
    Mdarray patches_2d = patches.View({B * num_patches_, patch_dim_});
    Mdarray embed = patch_embed_.Forward(patches_2d).View({B, num_patches_, d_model_});

    // Add positional encoding
    Mdarray x = embed + pe_;

    // Transformer blocks
    for (auto& blk : blocks_) {
      x = blk->Forward(x);
    }

    // Global average pooling -> (B, d_model)
    Mdarray pooled = Operator::CreateOperationMean(x, 1);

    // Classification head
    Mdarray logits = head_.Forward(pooled);
    return logits;
  }

  Learning::ParamsDict Parameters() override {
    Learning::ParamsDict dict = {
        {"patch_embed", patch_embed_.Parameters()},
        {"head", head_.Parameters()},
    };
    for (Index i = 0; i < blocks_.size(); ++i) {
      auto block_params = blocks_[i]->Parameters();
      for (auto& kv : block_params) {
        dict.insert({"block_" + std::to_string(i) + "_" + kv.first, kv.second});
      }
    }
    return dict;
  }

 private:
  void BuildSinusoidalPE() {
    pe_buf_.resize(num_patches_ * d_model_);
    for (Index pos = 0; pos < num_patches_; ++pos) {
      for (Index i = 0; i < d_model_; ++i) {
        double div = std::pow(10000.0, (i / 2) * 2.0 / static_cast<double>(d_model_));
        double val = pos / div;
        double v = (i % 2 == 0) ? std::sin(val) : std::cos(val);
        pe_buf_[pos * d_model_ + i] = static_cast<BasicData>(v);
      }
    }
    pe_ = Mdarray(pe_buf_.data(), Shape({1, num_patches_, d_model_}));
  }

  Index patch_size_;
  Index d_model_;
  Index n_heads_;
  Index d_ff_;
  Index n_layers_;
  Index num_patches_;
  Index patch_dim_;

  Learning::Linear patch_embed_;
  Learning::Linear head_;
  std::vector<std::unique_ptr<TransformerBlock>> blocks_;

  std::vector<BasicData> pe_buf_;
  Mdarray pe_;
};

//=============================================================================
// 训练和评估辅助函数
//=============================================================================

struct TrainConfig {
  std::string model_name;
  Index epochs = 5;
  Index batch_size = 64;
  BasicData lr = 0.001;
  BasicData momentum = 0.9;
  Index print_interval = 50;
  Index max_iters_per_epoch = 0;  // 0 表示不限制
};

struct TrainResult {
  std::string model_name;
  std::vector<BasicData> train_losses;
  std::vector<BasicData> val_accuracies;
  double train_time_seconds;
};

// 通用训练函数
template <typename ModelType>
TrainResult TrainModel(ModelType& model, const TrainConfig& config) {
  TrainResult result;
  result.model_name = config.model_name;

  using namespace std::chrono;
  steady_clock::time_point start_tp = steady_clock::now();

  // 加载数据集
  SourceData::Cifar10 train_dataset = 
      SourceData::Cifar10::CreateTrainDataset(config.batch_size);
  SourceData::Cifar10 val_dataset = 
      SourceData::Cifar10::CreateTestDataset(config.batch_size);

  // 损失函数和优化器
  Learning::CrossEntropy criterion;
  Learning::Adam optimizer(model.Parameters(), config.lr, 0.9, 0.999, 1e-8);

  LOG_MDA_INFO("========================================")
  LOG_MDA_INFO("Training " << config.model_name)
  LOG_MDA_INFO("Epochs: " << config.epochs << ", Batch size: " << config.batch_size)
  LOG_MDA_INFO("Learning rate: " << config.lr)
  LOG_MDA_INFO("========================================")

  Index samples_size;
  const BasicData* batch_samples;
  const Index* batch_labels;

  for (Index epoch = 0; epoch < config.epochs; ++epoch) {
    LOG_MDA_INFO("Epoch " << epoch << " training...")
    train_dataset.Shuffle();

    Index max_iters = train_dataset.BatchesSize();
    if (config.max_iters_per_epoch > 0) {
      max_iters = std::min(max_iters, config.max_iters_per_epoch);
    }

    BasicData epoch_loss = 0;
    Index loss_count = 0;

    for (Index j = 0; j < max_iters; ++j) {
      std::tie(samples_size, batch_samples, batch_labels) =
          train_dataset.GetBatch(j);
      
      Mdarray input(batch_samples,
                    {samples_size, SourceData::Cifar10::Img::channels_size_,
                     SourceData::Cifar10::Img::rows_size_,
                     SourceData::Cifar10::Img::cols_size_});

      Mdarray output = model.Forward(input);
      Mdarray loss = criterion.Forward(output, batch_labels);
      loss.Backward();

      optimizer.Step();
      optimizer.ZeroGrad();

      epoch_loss += loss.Item();
      ++loss_count;

      if (j % config.print_interval == 0) {
        LOG_MDA_INFO("iter " << j << " | loss: " << loss.Item())
      }
    }

    result.train_losses.push_back(epoch_loss / loss_count);

    // 评估
    LOG_MDA_INFO("Epoch " << epoch << " evaluating...")
    Index total_samples = 0, correct_samples = 0;
    
    Index val_max_iters = val_dataset.BatchesSize();
    if (config.max_iters_per_epoch > 0) {
      val_max_iters = std::min(val_max_iters, config.max_iters_per_epoch / 2);
    }

    for (Index j = 0; j < val_max_iters; ++j) {
      std::tie(samples_size, batch_samples, batch_labels) =
          val_dataset.GetBatch(j);
      
      Mdarray input(batch_samples,
                    {samples_size, SourceData::Cifar10::Img::channels_size_,
                     SourceData::Cifar10::Img::rows_size_,
                     SourceData::Cifar10::Img::cols_size_});

      Mdarray output = model.Forward(input);
      Mdarray predict = Operator::CreateOperationArgmax(output, 1);
      
      for (Index k = 0; k < samples_size; ++k) {
        ++total_samples;
        Index pd_label = static_cast<Index>(predict[{k}]);
        if (pd_label == batch_labels[k]) ++correct_samples;
      }
    }

    BasicData acc = static_cast<BasicData>(correct_samples) / total_samples;
    result.val_accuracies.push_back(acc);
    LOG_MDA_INFO("Accuracy: " << acc * 100 << "% (" << correct_samples << "/" << total_samples << ")")
  }

  steady_clock::time_point end_tp = steady_clock::now();
  duration<double> time_span = duration_cast<duration<double>>(end_tp - start_tp);
  result.train_time_seconds = time_span.count();

  LOG_MDA_INFO("Training " << config.model_name << " finished in " 
               << result.train_time_seconds << " seconds")

  return result;
}

//=============================================================================
// 主函数
//=============================================================================
int main(int argc, char** argv) {
  LOG_MDA_INFO("==========================================================")
  LOG_MDA_INFO("   CIFAR-10 Model Comparison: MLP vs CNN vs Transformer   ")
  LOG_MDA_INFO("==========================================================")

  // 配置
  TrainConfig mlp_config;
  mlp_config.model_name = "MLP";
  mlp_config.epochs = 3;
  mlp_config.batch_size = 64;
  mlp_config.lr = 0.001;
  mlp_config.print_interval = 50;
  mlp_config.max_iters_per_epoch = 200;  // 限制迭代数加速测试

  TrainConfig cnn_config;
  cnn_config.model_name = "CNN";
  cnn_config.epochs = 3;
  cnn_config.batch_size = 64;
  cnn_config.lr = 0.001;
  cnn_config.print_interval = 50;
  cnn_config.max_iters_per_epoch = 200;

  TrainConfig vit_config;
  vit_config.model_name = "ViT";
  vit_config.epochs = 3;
  vit_config.batch_size = 16;  // 更小的 batch
  vit_config.lr = 0.0005;
  vit_config.print_interval = 50;
  vit_config.max_iters_per_epoch = 150;  // 减少迭代数

  std::vector<TrainResult> results;

  // 训练 MLP
  {
    LOG_MDA_INFO("\n>>> Starting MLP Training <<<")
    MLP_Cifar10 mlp;
    results.push_back(TrainModel(mlp, mlp_config));
  }

  // 训练 CNN
  {
    LOG_MDA_INFO("\n>>> Starting CNN Training <<<")
    CNN_Cifar10 cnn;
    results.push_back(TrainModel(cnn, cnn_config));
  }

  // 训练 ViT
  {
    LOG_MDA_INFO("\n>>> Starting ViT Training <<<")
    // ViT 配置: patch_size=8 (更大patch减少序列长度), d_model=32, n_heads=4, d_ff=64, n_layers=1
    ViT_Cifar10 vit(8, 32, 4, 64, 1, 10);
    results.push_back(TrainModel(vit, vit_config));
  }

  // 打印对比结果
  LOG_MDA_INFO("\n==========================================================")
  LOG_MDA_INFO("                    COMPARISON RESULTS                     ")
  LOG_MDA_INFO("==========================================================")
  LOG_MDA_INFO("")
  LOG_MDA_INFO("Model       | Final Acc  | Best Acc   | Train Time ")
  LOG_MDA_INFO("------------|------------|------------|------------")

  for (const auto& r : results) {
    BasicData final_acc = r.val_accuracies.empty() ? 0 : r.val_accuracies.back();
    BasicData best_acc = 0;
    for (auto a : r.val_accuracies) best_acc = std::max(best_acc, a);

    char buf[256];
    snprintf(buf, sizeof(buf), "%-11s | %9.2f%% | %9.2f%% | %7.1f s",
             r.model_name.c_str(), final_acc * 100, best_acc * 100, r.train_time_seconds);
    LOG_MDA_INFO(buf)
  }

  LOG_MDA_INFO("")
  LOG_MDA_INFO("==========================================================")
  LOG_MDA_INFO("Analysis:")
  LOG_MDA_INFO("- CNN is best suited for image classification (spatial inductive bias)")
  LOG_MDA_INFO("- MLP lacks spatial awareness, needs more data/parameters")
  LOG_MDA_INFO("- ViT needs more data/epochs to match CNN on small datasets")
  LOG_MDA_INFO("==========================================================")

  return 0;
}
