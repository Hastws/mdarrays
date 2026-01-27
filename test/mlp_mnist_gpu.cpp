// GPU 版本的 MLP MNIST 训练
// 使用新的 GPU 后端架构

#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <random>
#include <tuple>

#include "backend/device.h"
#include "backend/gpu_tensor.h"
#include "backend/gpu_module.h"
#include "data/data.h"

using namespace Autoalg;
using namespace Autoalg::GPU;

int main() {
    std::cout << "=== GPU MLP MNIST Training ===" << std::endl;
    
    // 选择设备
#ifdef USE_CUDA
    DeviceType device = DeviceType::CUDA;
    std::cout << "Using CUDA device" << std::endl;
#else
    DeviceType device = DeviceType::CPU;
    std::cout << "Using CPU device" << std::endl;
#endif
    
    // 加载数据
    std::cout << "\nLoading MNIST data..." << std::endl;
    
    // 超参数
    const Index batch_size = 64;
    const Index input_size = 784;  // 28 * 28
    const Index hidden_size = 128;
    const Index num_classes = 10;
    const Index num_epochs = 3;
    const BasicData learning_rate = 0.01;
    
    // 数据集
    SourceData::MNIST train_dataset(std::string(MLP_MNIST_TRAIN_IMAGES),
                                    std::string(MLP_MNIST_TRAIN_LABELS),
                                    batch_size);
    SourceData::MNIST test_dataset(std::string(MLP_MNIST_TEST_IMAGES),
                                   std::string(MLP_MNIST_TEST_LABELS),
                                   batch_size);
    
    // 构建模型
    std::cout << "Building model..." << std::endl;
    Sequential model;
    model.Add<Linear>(input_size, hidden_size, device);
    model.Add<ReLU>();
    model.Add<Linear>(hidden_size, hidden_size, device);
    model.Add<ReLU>();
    model.Add<Linear>(hidden_size, num_classes, device);
    
    CrossEntropyLoss criterion;
    
    // 训练循环
    std::cout << "\nStarting training..." << std::endl;
    auto total_start = std::chrono::high_resolution_clock::now();
    
    for (Index epoch = 0; epoch < num_epochs; ++epoch) {
        std::cout << "\n--- Epoch " << epoch << " ---" << std::endl;
        
        // 训练
        BasicData total_loss = 0;
        Index num_batches = 0;
        Index correct = 0;
        Index total = 0;
        
        auto epoch_start = std::chrono::high_resolution_clock::now();
        train_dataset.Shuffle();
        
        Index max_train_batches = std::min(train_dataset.BatchesSize(), (Index)500);
        
        for (Index batch_idx = 0; batch_idx < max_train_batches; ++batch_idx) {
            Index n_samples;
            const BasicData* batch_samples;
            const Index* batch_labels;
            
            std::tie(n_samples, batch_samples, batch_labels) = train_dataset.GetBatch(batch_idx);
            
            // 准备输入数据
            GpuTensor input(batch_samples, Shape({n_samples, input_size}), device);
            std::vector<Index> targets(batch_labels, batch_labels + n_samples);
            
            // 前向传播
            model.ZeroGrad();
            GpuTensor output = model.Forward(input);
            
            // 计算损失
            BasicData loss = criterion.Forward(output, targets);
            total_loss += loss;
            
            // 反向传播
            GpuTensor grad = criterion.Backward();
            model.Backward(grad);
            
            // 更新参数
            model.UpdateParams(learning_rate);
            
            // 计算准确率
            std::vector<BasicData> out_data = output.ToVector();
            for (Index b = 0; b < n_samples; ++b) {
                Index pred = 0;
                BasicData max_val = out_data[b * num_classes];
                for (Index c = 1; c < num_classes; ++c) {
                    if (out_data[b * num_classes + c] > max_val) {
                        max_val = out_data[b * num_classes + c];
                        pred = c;
                    }
                }
                if (pred == targets[b]) correct++;
                total++;
            }
            
            num_batches++;
            
            if (num_batches % 100 == 0) {
                std::cout << "  Batch " << num_batches 
                          << " | Loss: " << total_loss / num_batches
                          << " | Acc: " << (100.0 * correct / total) << "%" << std::endl;
            }
        }
        
        auto epoch_end = std::chrono::high_resolution_clock::now();
        double epoch_time = std::chrono::duration<double>(epoch_end - epoch_start).count();
        
        std::cout << "Epoch " << epoch << " completed in " << epoch_time << "s"
                  << " | Avg Loss: " << total_loss / num_batches
                  << " | Train Acc: " << (100.0 * correct / total) << "%" << std::endl;
        
        // 验证
        correct = 0;
        total = 0;
        Index max_test_batches = std::min(test_dataset.BatchesSize(), (Index)100);
        
        for (Index batch_idx = 0; batch_idx < max_test_batches; ++batch_idx) {
            Index n_samples;
            const BasicData* batch_samples;
            const Index* batch_labels;
            
            std::tie(n_samples, batch_samples, batch_labels) = test_dataset.GetBatch(batch_idx);
            
            GpuTensor input(batch_samples, Shape({n_samples, input_size}), device);
            std::vector<Index> targets(batch_labels, batch_labels + n_samples);
            
            GpuTensor output = model.Forward(input);
            std::vector<BasicData> out_data = output.ToVector();
            
            for (Index b = 0; b < n_samples; ++b) {
                Index pred = 0;
                BasicData max_val = out_data[b * num_classes];
                for (Index c = 1; c < num_classes; ++c) {
                    if (out_data[b * num_classes + c] > max_val) {
                        max_val = out_data[b * num_classes + c];
                        pred = c;
                    }
                }
                if (pred == targets[b]) correct++;
                total++;
            }
        }
        
        std::cout << "Validation Acc: " << (100.0 * correct / total) << "% (" 
                  << correct << "/" << total << ")" << std::endl;
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(total_end - total_start).count();
    
    std::cout << "\n=== Training Complete ===" << std::endl;
    std::cout << "Total time: " << total_time << " seconds" << std::endl;
    
    return 0;
}
