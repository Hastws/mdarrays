#ifndef MULTIDIMENSIONAL_ARRAYS_INCLUDE_BACKEND_GPU_MODULE_H
#define MULTIDIMENSIONAL_ARRAYS_INCLUDE_BACKEND_GPU_MODULE_H

#include <vector>
#include <memory>
#include <cmath>
#include <random>
#include "backend/gpu_tensor.h"

namespace Autoalg {
namespace GPU {

// C++11 兼容的 make_unique
template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

//=============================================================================
// 基础模块接口
//=============================================================================

class Module {
public:
    virtual ~Module() = default;
    virtual GpuTensor Forward(const GpuTensor& input) = 0;
    virtual GpuTensor Backward(const GpuTensor& grad_output) = 0;
    virtual void UpdateParams(BasicData lr) {}
    virtual void ToDevice(DeviceType device) {}
    virtual void ZeroGrad() {}
};

//=============================================================================
// Linear 层
//=============================================================================

class Linear : public Module {
public:
    Linear(Index in_features, Index out_features, DeviceType device = DeviceType::CPU)
        : in_features_(in_features), out_features_(out_features), device_(device) {
        // Xavier 初始化
        BasicData std = std::sqrt(2.0 / (in_features + out_features));
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<BasicData> dist(0.0, std);
        
        std::vector<BasicData> w_data(in_features * out_features);
        std::vector<BasicData> b_data(out_features, 0.0);
        for (auto& v : w_data) v = dist(gen);
        
        weight_ = GpuTensor(w_data.data(), Shape({in_features, out_features}), device);
        bias_ = GpuTensor(b_data.data(), Shape({1, out_features}), device);
        
        grad_weight_ = GpuTensor(Shape({in_features, out_features}), 0.0, device);
        grad_bias_ = GpuTensor(Shape({1, out_features}), 0.0, device);
    }
    
    GpuTensor Forward(const GpuTensor& input) override {
        input_ = input;  // 保存用于反向传播
        Index batch = input.Size(0);
        
        // output = input @ weight + bias
        GpuTensor output(Shape({batch, out_features_}), device_);
        GpuTensor::Matmul(output, input, weight_);
        
        // 广播加 bias
        if (device_ == DeviceType::CPU) {
            for (Index b = 0; b < batch; ++b) {
                SIMD::add(output.Data() + b * out_features_,
                         output.Data() + b * out_features_,
                         bias_.Data(), out_features_);
            }
        } else {
#ifdef USE_CUDA
            // GPU 上用 kernel 处理广播
            for (Index b = 0; b < batch; ++b) {
                CUDA::add(output.Data() + b * out_features_,
                         output.Data() + b * out_features_,
                         bias_.Data(), out_features_);
            }
#endif
        }
        
        return output;
    }
    
    GpuTensor Backward(const GpuTensor& grad_output) override {
        Index batch = input_.Size(0);
        
        // grad_weight = input.T @ grad_output
        GpuTensor input_t(Shape({in_features_, batch}), device_);
        Transpose2D(input_t, input_);
        
        GpuTensor gw(Shape({in_features_, out_features_}), device_);
        GpuTensor::Matmul(gw, input_t, grad_output);
        GpuTensor::AddInplace(grad_weight_, gw);
        
        // grad_bias = sum(grad_output, dim=0)
        if (device_ == DeviceType::CPU) {
            for (Index b = 0; b < batch; ++b) {
                SIMD::add_inplace(grad_bias_.Data(), 
                                 grad_output.Data() + b * out_features_, 
                                 out_features_);
            }
        } else {
#ifdef USE_CUDA
            CUDA::col_sum(grad_bias_.Data(), grad_output.Data(), batch, out_features_);
#endif
        }
        
        // grad_input = grad_output @ weight.T
        GpuTensor weight_t(Shape({out_features_, in_features_}), device_);
        Transpose2D(weight_t, weight_);
        
        GpuTensor grad_input(Shape({batch, in_features_}), device_);
        GpuTensor::Matmul(grad_input, grad_output, weight_t);
        
        return grad_input;
    }
    
    void UpdateParams(BasicData lr) override {
        // weight -= lr * grad_weight
        if (device_ == DeviceType::CPU) {
            for (Index i = 0; i < weight_.TotalSize(); ++i) {
                weight_.Data()[i] -= lr * grad_weight_.Data()[i];
            }
            for (Index i = 0; i < bias_.TotalSize(); ++i) {
                bias_.Data()[i] -= lr * grad_bias_.Data()[i];
            }
        } else {
#ifdef USE_CUDA
            // GPU: weight = weight - lr * grad
            GpuTensor scaled_grad(grad_weight_.Size(), device_);
            GpuTensor::Scale(scaled_grad, grad_weight_, lr);
            GpuTensor::Sub(weight_, weight_, scaled_grad);
            
            GpuTensor scaled_bias_grad(grad_bias_.Size(), device_);
            GpuTensor::Scale(scaled_bias_grad, grad_bias_, lr);
            GpuTensor::Sub(bias_, bias_, scaled_bias_grad);
#endif
        }
    }
    
    void ZeroGrad() override {
        grad_weight_.FillZero();
        grad_bias_.FillZero();
    }
    
    void ToDevice(DeviceType device) override {
        if (device_ != device) {
            weight_ = weight_.To(device);
            bias_ = bias_.To(device);
            grad_weight_ = grad_weight_.To(device);
            grad_bias_ = grad_bias_.To(device);
            device_ = device;
        }
    }

private:
    static void Transpose2D(GpuTensor& dst, const GpuTensor& src) {
        Index rows = src.Size(0);
        Index cols = src.Size(1);
        
        if (src.IsCpu()) {
            for (Index i = 0; i < rows; ++i) {
                for (Index j = 0; j < cols; ++j) {
                    dst.Data()[j * rows + i] = src.Data()[i * cols + j];
                }
            }
        } else {
#ifdef USE_CUDA
            // 简单实现：先拷贝到 CPU 做转置，再拷回 GPU
            // TODO: 实现 CUDA transpose kernel
            std::vector<BasicData> src_data(rows * cols);
            std::vector<BasicData> dst_data(rows * cols);
            src.Storage().CopyToHost(src_data.data(), rows * cols);
            
            for (Index i = 0; i < rows; ++i) {
                for (Index j = 0; j < cols; ++j) {
                    dst_data[j * rows + i] = src_data[i * cols + j];
                }
            }
            
            dst.Storage().CopyFromHost(dst_data.data(), rows * cols);
#endif
        }
    }
    
    Index in_features_;
    Index out_features_;
    DeviceType device_;
    
    GpuTensor weight_;
    GpuTensor bias_;
    GpuTensor grad_weight_;
    GpuTensor grad_bias_;
    GpuTensor input_;  // 缓存用于反向传播
};

//=============================================================================
// ReLU 激活
//=============================================================================

class ReLU : public Module {
public:
    GpuTensor Forward(const GpuTensor& input) override {
        input_ = input;
        GpuTensor output(input.Size(), input.Device());
        GpuTensor::ReLU(output, input);
        return output;
    }
    
    GpuTensor Backward(const GpuTensor& grad_output) override {
        GpuTensor grad_input(input_.Size(), input_.Device());
        
        if (input_.IsCpu()) {
            SIMD::relu_backward(grad_input.Data(), grad_output.Data(), 
                               input_.Data(), input_.TotalSize());
        } else {
#ifdef USE_CUDA
            CUDA::relu_backward(grad_input.Data(), grad_output.Data(),
                               input_.Data(), input_.TotalSize());
#endif
        }
        
        return grad_input;
    }

private:
    GpuTensor input_;
};

//=============================================================================
// Softmax
//=============================================================================

class Softmax : public Module {
public:
    GpuTensor Forward(const GpuTensor& input) override {
        output_ = GpuTensor(input.Size(), input.Device());
        GpuTensor::Softmax(output_, input);
        return output_;
    }
    
    GpuTensor Backward(const GpuTensor& grad_output) override {
        // Softmax 反向传播：grad_input = output * (grad_output - sum(grad_output * output))
        Index batch = output_.Size(0);
        Index dim = output_.Size(1);
        GpuTensor grad_input(output_.Size(), output_.Device());
        
        if (output_.IsCpu()) {
            for (Index b = 0; b < batch; ++b) {
                BasicData* gi = grad_input.Data() + b * dim;
                const BasicData* go = grad_output.Data() + b * dim;
                const BasicData* o = output_.Data() + b * dim;
                
                // sum(grad_output * output)
                BasicData dot_sum = SIMD::dot(go, o, dim);
                
                // grad_input = output * (grad_output - dot_sum)
                for (Index i = 0; i < dim; ++i) {
                    gi[i] = o[i] * (go[i] - dot_sum);
                }
            }
        } else {
#ifdef USE_CUDA
            // GPU 版本 - 简化实现
            std::vector<BasicData> go_data(batch * dim);
            std::vector<BasicData> o_data(batch * dim);
            std::vector<BasicData> gi_data(batch * dim);
            
            grad_output.Storage().CopyToHost(go_data.data(), batch * dim);
            output_.Storage().CopyToHost(o_data.data(), batch * dim);
            
            for (Index b = 0; b < batch; ++b) {
                BasicData dot_sum = 0;
                for (Index i = 0; i < dim; ++i) {
                    dot_sum += go_data[b * dim + i] * o_data[b * dim + i];
                }
                for (Index i = 0; i < dim; ++i) {
                    gi_data[b * dim + i] = o_data[b * dim + i] * (go_data[b * dim + i] - dot_sum);
                }
            }
            
            grad_input.Storage().CopyFromHost(gi_data.data(), batch * dim);
#endif
        }
        
        return grad_input;
    }

private:
    GpuTensor output_;
};

//=============================================================================
// Sequential 容器
//=============================================================================

class Sequential : public Module {
public:
    Sequential() = default;
    
    template<typename T, typename... Args>
    void Add(Args&&... args) {
        modules_.push_back(GPU::make_unique<T>(std::forward<Args>(args)...));
    }
    
    void Add(std::unique_ptr<Module> module) {
        modules_.push_back(std::move(module));
    }
    
    GpuTensor Forward(const GpuTensor& input) override {
        GpuTensor x = input;
        for (auto& m : modules_) {
            x = m->Forward(x);
        }
        return x;
    }
    
    GpuTensor Backward(const GpuTensor& grad_output) override {
        GpuTensor grad = grad_output;
        for (int i = static_cast<int>(modules_.size()) - 1; i >= 0; --i) {
            grad = modules_[i]->Backward(grad);
        }
        return grad;
    }
    
    void UpdateParams(BasicData lr) override {
        for (auto& m : modules_) {
            m->UpdateParams(lr);
        }
    }
    
    void ZeroGrad() override {
        for (auto& m : modules_) {
            m->ZeroGrad();
        }
    }
    
    void ToDevice(DeviceType device) override {
        for (auto& m : modules_) {
            m->ToDevice(device);
        }
    }

private:
    std::vector<std::unique_ptr<Module>> modules_;
};

//=============================================================================
// 损失函数
//=============================================================================

class CrossEntropyLoss {
public:
    // input: [batch, num_classes] (logits)
    // target: [batch] (class indices)
    BasicData Forward(const GpuTensor& input, const std::vector<Index>& target) {
        batch_ = input.Size(0);
        num_classes_ = input.Size(1);
        
        // Softmax
        probs_ = GpuTensor(input.Size(), input.Device());
        GpuTensor::Softmax(probs_, input);
        
        // 计算交叉熵损失
        std::vector<BasicData> probs_data = probs_.ToVector();
        BasicData loss = 0;
        for (Index b = 0; b < batch_; ++b) {
            BasicData p = probs_data[b * num_classes_ + target[b]];
            loss -= std::log(std::max(p, 1e-12));
        }
        
        target_ = target;
        return loss / batch_;
    }
    
    GpuTensor Backward() {
        // grad = probs - one_hot(target)
        std::vector<BasicData> grad_data = probs_.ToVector();
        for (Index b = 0; b < batch_; ++b) {
            grad_data[b * num_classes_ + target_[b]] -= 1.0;
        }
        
        // 除以 batch size
        for (auto& g : grad_data) {
            g /= batch_;
        }
        
        GpuTensor grad(grad_data.data(), Shape({batch_, num_classes_}), probs_.Device());
        return grad;
    }

private:
    Index batch_;
    Index num_classes_;
    GpuTensor probs_;
    std::vector<Index> target_;
};

}  // namespace GPU
}  // namespace Autoalg

#endif  // MULTIDIMENSIONAL_ARRAYS_INCLUDE_BACKEND_GPU_MODULE_H
