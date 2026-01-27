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
    static DeviceType& CurrentDevice() {
        static DeviceType device = DeviceType::CPU;
        return device;
    }
    
    static int& CudaDeviceId() {
        static int device_id = 0;
        return device_id;
    }
    
    static void SetDevice(DeviceType type, int device_id = 0) {
        CurrentDevice() = type;
        CudaDeviceId() = device_id;
    }
    
    static DeviceType GetDevice() { return CurrentDevice(); }
    static int GetCudaDeviceId() { return CudaDeviceId(); }
    
    static bool IsCuda() { return CurrentDevice() == DeviceType::CUDA; }
    static bool IsCpu() { return CurrentDevice() == DeviceType::CPU; }
    
    static std::string DeviceName() {
        switch (CurrentDevice()) {
            case DeviceType::CUDA: return "cuda:" + std::to_string(CudaDeviceId());
            default: return "cpu";
        }
    }
};

}  // namespace Autoalg

#endif  // MULTIDIMENSIONAL_ARRAYS_INCLUDE_BACKEND_DEVICE_H
