#ifndef MULTIDIMENSIONAL_ARRAYS_INCLUDE_MDARRAY_STORAGE_H
#define MULTIDIMENSIONAL_ARRAYS_INCLUDE_MDARRAY_STORAGE_H

#include <cstring>
#include <memory>

#include "memory_pool/allocator.h"
#include "utils/base_config.h"
#include "utils/log.h"
#include "backend/simd_kernel.h"

namespace Autoalg {

class StorageUniversalAgent;

class Storage {
 public:
  explicit Storage(Index size)
      : size_(size),
        base_ptr_(Allocator::SharedAllocate<VersionData>(
            size * sizeof(BasicData) + sizeof(Index) + sizeof(Index))),
        data_ptr_(base_ptr_->data_) {
    base_ptr_->version_ = 0;
    base_ptr_->size_ = size;
  }

  Storage(const Storage &other, Index offset)
      : size_(other.size_ - offset),
        base_ptr_(other.base_ptr_), 
        data_ptr_(other.data_ptr_ + offset) {}

  Storage(Index size, BasicData value) : Storage(size) {
    SIMD::fill_const(data_ptr_, value, size);
  }

  Storage(const BasicData *data, Index size) : Storage(size) {
    SIMD::copy(data_ptr_, data, size);
  }

  Storage(const Storage &other) = default;
  Storage(Storage &&other) = default;
  ~Storage() = default;
  Storage &operator=(const Storage &other) = delete;

  // inline function
  BasicData operator[](Index idx) const { return data_ptr_[idx]; }
  BasicData &operator[](Index idx) { return data_ptr_[idx]; }

  Index Offset() const { return data_ptr_ - base_ptr_->data_; }
  Index Version() const { return base_ptr_->version_; }
  void IncrementVersion() const { ++base_ptr_->version_; }
  
  // 批量操作接口
  BasicData* Data() { return data_ptr_; }
  const BasicData* Data() const { return data_ptr_; }
  Index Size() const { return size_; }

  // friend function
  friend class StorageUniversalAgent;

 private:
  struct VersionData {
    Index version_;
    Index size_;
    BasicData data_[1];
  };

  Index size_;
  std::shared_ptr<VersionData> base_ptr_;  // base pointer
  BasicData *data_ptr_;                    // SourceData pointer
};

class StorageUniversalAgent {
 public:
  explicit StorageUniversalAgent(const Storage &storage) : storage_(storage) {}
  BasicData *GetStorageData() const { return storage_.data_ptr_; }

 private:
  const Storage &storage_;
};

}  // namespace Autoalg
#endif