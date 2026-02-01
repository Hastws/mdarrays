#ifndef MULTIDIMENSIONAL_ARRAYS_INCLUDE_DATA_DATA_DOWNLOADER_H
#define MULTIDIMENSIONAL_ARRAYS_INCLUDE_DATA_DATA_DOWNLOADER_H

#include <string>

namespace Autoalg {
namespace SourceData {

/**
 * @brief 数据下载器 - 负责下载和解压数据集
 * 
 * 支持的数据集:
 * - MNIST: 手写数字识别数据集
 * - CIFAR-10: 10类图像分类数据集
 */
class DataDownloader {
 public:
  /**
   * @brief 获取数据根目录
   * @return 数据目录路径，默认为可执行文件所在目录的 ../data 或环境变量 MDARRAYS_DATA_DIR
   */
  static std::string GetDataRoot();
  
  /**
   * @brief 设置数据根目录
   * @param path 数据目录路径
   */
  static void SetDataRoot(const std::string& path);

  /**
   * @brief 确保MNIST数据集已下载
   * @return MNIST数据目录路径
   */
  static std::string EnsureMNIST();

  /**
   * @brief 确保CIFAR-10数据集已下载
   * @return CIFAR-10数据目录路径
   */
  static std::string EnsureCifar10();

  /**
   * @brief 获取MNIST训练图像路径
   */
  static std::string GetMNISTTrainImages();

  /**
   * @brief 获取MNIST训练标签路径
   */
  static std::string GetMNISTTrainLabels();

  /**
   * @brief 获取MNIST测试图像路径
   */
  static std::string GetMNISTTestImages();

  /**
   * @brief 获取MNIST测试标签路径
   */
  static std::string GetMNISTTestLabels();

  /**
   * @brief 获取CIFAR-10数据目录路径
   */
  static std::string GetCifar10Dir();

 private:
  /**
   * @brief 下载文件
   * @param url 文件URL
   * @param output_path 输出路径
   * @return 是否成功
   */
  static bool DownloadFile(const std::string& url, const std::string& output_path);

  /**
   * @brief 解压gzip文件
   * @param gz_path gzip文件路径
   * @param output_path 解压输出路径
   * @return 是否成功
   */
  static bool DecompressGzip(const std::string& gz_path, const std::string& output_path);

  /**
   * @brief 解压tar.gz文件
   * @param tar_gz_path tar.gz文件路径
   * @param output_dir 解压输出目录
   * @return 是否成功
   */
  static bool ExtractTarGz(const std::string& tar_gz_path, const std::string& output_dir);

  /**
   * @brief 创建目录（递归）
   * @param path 目录路径
   * @return 是否成功
   */
  static bool CreateDirectoryRecursive(const std::string& path);

  /**
   * @brief 检查文件是否存在
   * @param path 文件路径
   * @return 是否存在
   */
  static bool FileExists(const std::string& path);

  /**
   * @brief 获取可执行文件所在目录
   * @return 目录路径
   */
  static std::string GetExecutableDir();

  static std::string data_root_;
};

}  // namespace SourceData
}  // namespace Autoalg

#endif
