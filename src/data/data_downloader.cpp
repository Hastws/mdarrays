#include "data/data_downloader.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>
#include <limits.h>

#include "utils/log.h"

namespace Autoalg {
namespace SourceData {

// 静态成员初始化
std::string DataDownloader::data_root_ = "";

// MNIST数据集URL - 使用 ossci-datasets 镜像 (PyTorch 官方使用的镜像)
static const char* MNIST_TRAIN_IMAGES_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz";
static const char* MNIST_TRAIN_LABELS_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz";
static const char* MNIST_TEST_IMAGES_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz";
static const char* MNIST_TEST_LABELS_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz";

// CIFAR-10数据集URL
static const char* CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz";

std::string DataDownloader::GetDataRoot() {
  if (!data_root_.empty()) {
    return data_root_;
  }

  // 首先检查环境变量
  const char* env_data_dir = std::getenv("MDARRAYS_DATA_DIR");
  if (env_data_dir != nullptr && env_data_dir[0] != '\0') {
    data_root_ = env_data_dir;
    return data_root_;
  }

  // 默认使用可执行文件所在目录的父目录下的 data 文件夹
  std::string exe_dir = GetExecutableDir();
  if (!exe_dir.empty()) {
    // 查找父目录
    size_t pos = exe_dir.rfind('/');
    if (pos != std::string::npos) {
      data_root_ = exe_dir.substr(0, pos) + "/data";
    } else {
      data_root_ = "./data";
    }
  } else {
    data_root_ = "./data";
  }

  return data_root_;
}

void DataDownloader::SetDataRoot(const std::string& path) {
  data_root_ = path;
}

std::string DataDownloader::GetExecutableDir() {
  char result[PATH_MAX];
  ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
  if (count == -1) {
    return "";
  }
  std::string path(result, count);
  size_t pos = path.rfind('/');
  if (pos != std::string::npos) {
    return path.substr(0, pos);
  }
  return "";
}

bool DataDownloader::FileExists(const std::string& path) {
  struct stat buffer;
  return (stat(path.c_str(), &buffer) == 0);
}

bool DataDownloader::CreateDirectoryRecursive(const std::string& path) {
  std::string current_path;
  std::istringstream path_stream(path);
  std::string segment;

  // 处理绝对路径
  if (path[0] == '/') {
    current_path = "/";
  }

  while (std::getline(path_stream, segment, '/')) {
    if (segment.empty()) continue;
    current_path += segment + "/";
    
    struct stat st;
    if (stat(current_path.c_str(), &st) != 0) {
      if (mkdir(current_path.c_str(), 0755) != 0) {
        LOG_MDA_ERROR("Failed to create directory: " << current_path);
        return false;
      }
    }
  }
  return true;
}

bool DataDownloader::DownloadFile(const std::string& url, const std::string& output_path) {
  LOG_MDA_INFO("Downloading: " << url);
  LOG_MDA_INFO("To: " << output_path);

  // 使用curl或wget下载
  std::ostringstream cmd;
  
  // 首先尝试curl (使用 -f 选项在服务器错误时失败，-s 静默模式，-S 显示错误)
  if (system("which curl > /dev/null 2>&1") == 0) {
    cmd << "curl -fsSL --retry 3 --retry-delay 2 -o \"" << output_path << "\" \"" << url << "\"";
  }
  // 然后尝试wget
  else if (system("which wget > /dev/null 2>&1") == 0) {
    cmd << "wget --tries=3 --waitretry=2 -q -O \"" << output_path << "\" \"" << url << "\"";
  }
  else {
    LOG_MDA_ERROR("Neither curl nor wget is available. Please install one of them.");
    return false;
  }

  int result = system(cmd.str().c_str());
  if (result != 0) {
    LOG_MDA_ERROR("Download failed with exit code: " << result);
    // 删除可能部分下载的文件
    std::remove(output_path.c_str());
    return false;
  }

  if (!FileExists(output_path)) {
    LOG_MDA_ERROR("Download completed but file not found: " << output_path);
    return false;
  }

  // 检查文件大小是否合理（至少1KB）
  struct stat st;
  if (stat(output_path.c_str(), &st) == 0 && st.st_size < 1024) {
    LOG_MDA_ERROR("Downloaded file is too small, likely an error page: " << output_path);
    std::remove(output_path.c_str());
    return false;
  }

  LOG_MDA_INFO("Download completed successfully.");
  return true;
}

bool DataDownloader::DecompressGzip(const std::string& gz_path, const std::string& output_path) {
  LOG_MDA_INFO("Decompressing: " << gz_path);

  std::ostringstream cmd;
  cmd << "gunzip -c \"" << gz_path << "\" > \"" << output_path << "\"";

  int result = system(cmd.str().c_str());
  if (result != 0) {
    LOG_MDA_ERROR("Decompression failed with exit code: " << result);
    return false;
  }

  if (!FileExists(output_path)) {
    LOG_MDA_ERROR("Decompression completed but file not found: " << output_path);
    return false;
  }

  // 删除.gz文件
  std::remove(gz_path.c_str());

  LOG_MDA_INFO("Decompression completed successfully.");
  return true;
}

bool DataDownloader::ExtractTarGz(const std::string& tar_gz_path, const std::string& output_dir) {
  LOG_MDA_INFO("Extracting: " << tar_gz_path);
  LOG_MDA_INFO("To: " << output_dir);

  std::ostringstream cmd;
  cmd << "tar -xzf \"" << tar_gz_path << "\" -C \"" << output_dir << "\"";

  int result = system(cmd.str().c_str());
  if (result != 0) {
    LOG_MDA_ERROR("Extraction failed with exit code: " << result);
    return false;
  }

  // 删除.tar.gz文件
  std::remove(tar_gz_path.c_str());

  LOG_MDA_INFO("Extraction completed successfully.");
  return true;
}

std::string DataDownloader::EnsureMNIST() {
  std::string data_root = GetDataRoot();
  std::string mnist_dir = data_root + "/mnist";

  // 检查目录是否存在
  if (!FileExists(mnist_dir)) {
    CreateDirectoryRecursive(mnist_dir);
  }

  // 检查所有文件是否存在
  std::string train_images = mnist_dir + "/train-images.idx3-ubyte";
  std::string train_labels = mnist_dir + "/train-labels.idx1-ubyte";
  std::string test_images = mnist_dir + "/t10k-images.idx3-ubyte";
  std::string test_labels = mnist_dir + "/t10k-labels.idx1-ubyte";

  bool all_exist = FileExists(train_images) && FileExists(train_labels) &&
                   FileExists(test_images) && FileExists(test_labels);

  if (all_exist) {
    LOG_MDA_INFO("MNIST data already exists at: " << mnist_dir);
    return mnist_dir;
  }

  LOG_MDA_INFO("MNIST data not found. Downloading...");

  // 下载并解压每个文件
  struct FileInfo {
    const char* url;
    std::string gz_path;
    std::string final_path;
  };

  FileInfo files[] = {
    {MNIST_TRAIN_IMAGES_URL, mnist_dir + "/train-images-idx3-ubyte.gz", train_images},
    {MNIST_TRAIN_LABELS_URL, mnist_dir + "/train-labels-idx1-ubyte.gz", train_labels},
    {MNIST_TEST_IMAGES_URL, mnist_dir + "/t10k-images-idx3-ubyte.gz", test_images},
    {MNIST_TEST_LABELS_URL, mnist_dir + "/t10k-labels-idx1-ubyte.gz", test_labels}
  };

  for (const auto& file : files) {
    if (FileExists(file.final_path)) {
      continue;
    }

    if (!DownloadFile(file.url, file.gz_path)) {
      LOG_MDA_ERROR("Failed to download MNIST data from: " << file.url);
      return "";
    }

    if (!DecompressGzip(file.gz_path, file.final_path)) {
      LOG_MDA_ERROR("Failed to decompress: " << file.gz_path);
      return "";
    }
  }

  LOG_MDA_INFO("MNIST data downloaded successfully to: " << mnist_dir);
  return mnist_dir;
}

std::string DataDownloader::EnsureCifar10() {
  std::string data_root = GetDataRoot();
  std::string cifar_dir = data_root + "/cifar_10";

  // 检查目录是否存在
  if (!FileExists(cifar_dir)) {
    CreateDirectoryRecursive(cifar_dir);
  }

  // 检查bin文件是否存在
  std::string test_batch = cifar_dir + "/test_batch.bin";
  std::string data_batch_1 = cifar_dir + "/data_batch_1.bin";

  bool all_exist = FileExists(test_batch) && FileExists(data_batch_1);

  if (all_exist) {
    LOG_MDA_INFO("CIFAR-10 data already exists at: " << cifar_dir);
    return cifar_dir;
  }

  LOG_MDA_INFO("CIFAR-10 data not found. Downloading...");

  std::string tar_gz_path = data_root + "/cifar-10-binary.tar.gz";

  if (!DownloadFile(CIFAR10_URL, tar_gz_path)) {
    LOG_MDA_ERROR("Failed to download CIFAR-10 data.");
    return "";
  }

  if (!ExtractTarGz(tar_gz_path, data_root)) {
    LOG_MDA_ERROR("Failed to extract CIFAR-10 data.");
    return "";
  }

  // CIFAR-10 解压后会在 cifar-10-batches-bin 目录，需要移动文件
  std::string extracted_dir = data_root + "/cifar-10-batches-bin";
  if (FileExists(extracted_dir)) {
    // 移动文件到 cifar_10 目录
    std::ostringstream cmd;
    cmd << "mv \"" << extracted_dir << "\"/* \"" << cifar_dir << "/\" 2>/dev/null; "
        << "rmdir \"" << extracted_dir << "\" 2>/dev/null";
    system(cmd.str().c_str());
  }

  if (FileExists(test_batch)) {
    LOG_MDA_INFO("CIFAR-10 data downloaded successfully to: " << cifar_dir);
    return cifar_dir;
  } else {
    LOG_MDA_ERROR("CIFAR-10 extraction completed but files not found.");
    return "";
  }
}

std::string DataDownloader::GetMNISTTrainImages() {
  std::string mnist_dir = EnsureMNIST();
  if (mnist_dir.empty()) {
    return "";
  }
  return mnist_dir + "/train-images.idx3-ubyte";
}

std::string DataDownloader::GetMNISTTrainLabels() {
  std::string mnist_dir = EnsureMNIST();
  if (mnist_dir.empty()) {
    return "";
  }
  return mnist_dir + "/train-labels.idx1-ubyte";
}

std::string DataDownloader::GetMNISTTestImages() {
  std::string mnist_dir = EnsureMNIST();
  if (mnist_dir.empty()) {
    return "";
  }
  return mnist_dir + "/t10k-images.idx3-ubyte";
}

std::string DataDownloader::GetMNISTTestLabels() {
  std::string mnist_dir = EnsureMNIST();
  if (mnist_dir.empty()) {
    return "";
  }
  return mnist_dir + "/t10k-labels.idx1-ubyte";
}

std::string DataDownloader::GetCifar10Dir() {
  return EnsureCifar10();
}

}  // namespace SourceData
}  // namespace Autoalg
