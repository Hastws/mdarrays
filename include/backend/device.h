#ifndef MULTIDIMENSIONAL_ARRAYS_INCLUDE_BACKEND_DEVICE_H
#define MULTIDIMENSIONAL_ARRAYS_INCLUDE_BACKEND_DEVICE_H

#include <string>
#include "utils/base_config.h"

namespace Autoalg {

enum class DeviceType {
    CPU = 0,
    CUDA = 1
};

class Device {
public:
    static DeviceType current_device;
    static int cuda_device_id;
    
    static void SetDevice(DeviceType type, int device_id = 0) {
        current_device = type;
        cuda_device_id = device_id;
    }
    
    static DeviceType GetDevice() { return current_device; }
    static int GetCudaDeviceId() { return cuda_device_id; }
    
    static bool IsCuda() { return current_device == DeviceType::CUDA; }
    static bool IsCpu() { return current_device == DeviceType::CPU; }
    
    static std::string DeviceName() {
        switch (current_device) {
            case DeviceType::CUDA: return "cuda:" + std::to_string(cuda_device_id);
            default: return "cpu";
        }
    }
};

// 静态成员初始化
inline DeviceType Device::current_device = DeviceType::CPU;
inline int Device::cuda_device_id = 0;

}  // namespace Autoalg

#endif  // MULTIDIMENSIONAL_ARRAYS_INCLUDE_BACKEND_DEVICE_H
