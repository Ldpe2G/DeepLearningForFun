#ifndef PARSE_MX_ND_UTILS_HPP
#define PARSE_MX_ND_UTILS_HPP

#include <stdio.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <dmlc/base.h>
#include <dmlc/logging.h>
#include <dmlc/io.h>
#include <dmlc/memory_io.h>
#include <map>
#include <string>

namespace parsend {
// Lowest NDArray object
// Can be regard as simple interpreter of void* 
// point.
class NDArray {
public:
  NDArray() {}
  ~NDArray() {}

  void* data = NULL;
  uint32_t numOfBytes = 0;
  std::string name = "";
  std::string type = "";
  std::vector<int64_t> shape;
};

enum NDArrayStorageType {
  kUndefinedStorage = -1,  // undefined storage
  kDefaultStorage,         // dense
  kRowSparseStorage,       // row sparse
  kCSRStorage,             // csr
};

size_t num_aux_data(NDArrayStorageType stype) {
  size_t num = 0;
  switch (stype) {
    case kDefaultStorage: num = 0; break;
    case kCSRStorage: LOG(FATAL) << "Not supported storage type" << stype; break; //num = 2; break;
    case kRowSparseStorage: LOG(FATAL) << "Not supported storage type" << stype; break; //num = 1; break;
     default: LOG(FATAL) << "Unknown storage type" << stype; break;
  }
  return num;
}

struct cpu {
  /*! \brief whether this device is CPU or not */
  static const bool kDevCPU = true;
  /*! \brief device flag number, identifies this device */
  static const int kDevMask = 1 << 0;
};
/*! \brief device name GPU */
struct gpu {
  /*! \brief whether this device is CPU or not */
  static const bool kDevCPU = false;
  /*! \brief device flag number, identifies this device */
  static const int kDevMask = 1 << 1;
};

/*! \brief Type of device */
enum DeviceType {
  kCPU = cpu::kDevMask,
  kGPU = gpu::kDevMask,
  kCPUPinned = 3,
  kCPUShared = 5,
};

/*! \brief data type flag */
enum TypeFlag {
  kFloat32 = 0,
  kFloat64 = 1,
  kFloat16 = 2,
  kUint8 = 3,
  kInt32 = 4,
  kInt8  = 5,
  kInt64 = 6,
};

std::string type2str(int type) {
  std::string typeStr;
  switch(type) {
    case kFloat32: typeStr = "float32"; break;
    case kFloat64: typeStr = "float64"; break;
    case kFloat16: typeStr = "float16"; break;
    case kUint8: typeStr = "uint8"; break;
    case kInt32: typeStr = "int32"; break;
    case kInt8: typeStr = "int8"; break;
    case kInt64: typeStr = "int64"; break;
    default:
      LOG(FATAL) << "Unknown data type";
  }
  return typeStr;
}

#define MSHADOW_TYPE_SWITCH(type, DType, ...)       \
  switch (type) {                                   \
  case kFloat32:                           \
    {                                               \
      typedef float DType;                          \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case kFloat64:                           \
    {                                               \
      typedef double DType;                         \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case kFloat16:                           \
    {                                               \
      typedef uint16_t DType;          \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case kUint8:                             \
    {                                               \
      typedef uint8_t DType;                        \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case kInt8:                              \
    {                                               \
      typedef int8_t DType;                         \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case kInt32:                             \
    {                                               \
      typedef int32_t DType;                        \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case kInt64:                             \
    {                                               \
      typedef int64_t DType;                        \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  default:                                          \
    LOG(FATAL) << "Unknown type enum " << type;     \
  }

/*! \brief get data type size from type enum */
inline size_t mshadow_sizeof(int type) {
  int size = 0;
  MSHADOW_TYPE_SWITCH(type, DType, size = sizeof(DType););
  return size;
}

static constexpr size_t alignment_ = 16;

const uint64_t kMXAPINDArrayListMagic = 0x112;
/* magic number for ndarray version 2, with storage type */
const uint32_t NDARRAY_V2_MAGIC = 0xF993fac9;

/*! \brief data type flag */
enum ReturnType {
  kSuccess = 0,
  kFailure = 1
};

int32_t loadNDArray(std::vector<NDArray *>& ndarrays, std::string param_file) {

  dmlc::Stream* fi = dmlc::Stream::Create(param_file.c_str(), "r");

  uint64_t header, reserved;
  if (!(fi->Read(&header))) {
    LOG(INFO) << "Invalid NDArray file format";
    return kFailure;
  }
  if(!(fi->Read(&reserved))) {
    LOG(INFO) << "Invalid NDArray file format";
    return kFailure;
  }
  if (header != kMXAPINDArrayListMagic) {
    LOG(INFO) << "Invalid NDArray file format";
    return kFailure;
  }

  uint64_t nd_size;
  if (fi->Read(&nd_size, sizeof(nd_size)) != sizeof(nd_size)) {
    LOG(INFO) << "read param error.";
    return kFailure;
  }

  size_t size = static_cast<size_t>(nd_size);
  
  ndarrays.resize(size);

  // read nd data
  for (size_t i = 0; i < nd_size; ++i) {
    NDArray* nd = new NDArray;
    ndarrays[i] = nd;

    uint32_t magic;
    if (fi->Read(&magic, sizeof(uint32_t)) != sizeof(uint32_t)) {
      LOG(INFO) << "read param error.";
      return kFailure;
    }
#if defined(DEBUG)
    LOG(INFO) << "magic code: " << magic;
#endif
    if (magic != NDARRAY_V2_MAGIC) {
      LOG(INFO) << "parsing ndarray with V1 format, not support yet!!";
      return kFailure;
    }

    // load storage type
    int32_t stype;
    if (fi->Read(&stype, sizeof(stype)) != sizeof(stype)) {
      LOG(INFO) << "read param error.";
      return kFailure;
    }

    const int32_t nad = num_aux_data(static_cast<NDArrayStorageType>(stype));
#if defined(DEBUG)
    LOG(INFO) << "storage type: " << nad;
#endif

    // load shape
    uint32_t ndim_{0};
    if (fi->Read(&ndim_, sizeof(ndim_)) != sizeof(ndim_)) {
      LOG(INFO) << "read param error.";
      return kFailure;
    }

    size_t nread = sizeof(int64_t) * ndim_;
#if defined(DEBUG)
    LOG(INFO) << "shape dim: " << ndim_ << ", nread: " << nread;
#endif
    int64_t *data_heap_ = new int64_t[ndim_];
    if (fi->Read(data_heap_, nread) != nread) {
      LOG(INFO) << "read param error.";
      return kFailure;
    }

    int64_t size = 1;
    for (uint32_t i=0; i<ndim_;++i) {
      size *= data_heap_[i];
      nd->shape.push_back(data_heap_[i]);
      // nd->shape += std::to_string(data_heap_[i]);
      // if (i < ndim_ - 1) nd->shape += "x";
    }

    delete[] data_heap_;

    // load context 
    DeviceType dev_type;
    int32_t dev_id;
    if (fi->Read(&dev_type, sizeof(dev_type)) != sizeof(dev_type)) {
      LOG(INFO) << "read param error.";
      return kFailure;
    }

    if (fi->Read(&dev_id, sizeof(int32_t)) != sizeof(int32_t)) {
      LOG(INFO) << "read param error.";
      return kFailure;
    }

#if defined(DEBUG)
    LOG(INFO) << "dev type: " << dev_type << ", dev id: " << dev_id;
#endif

    // load type flag
    int32_t type_flag;
    if (fi->Read(&type_flag, sizeof(type_flag)) != sizeof(type_flag)) {
      LOG(INFO) << "read param error.";
      return kFailure;
    }
#if defined(DEBUG)
    LOG(INFO) << "type flag: " << type_flag;
#endif
    nd->type = type2str(type_flag);

    size_t all_size = size * mshadow_sizeof(type_flag);

    nd->numOfBytes = all_size;
    // int ret = posix_memalign(&(nd->data), alignment_, all_size);
    nd->data = (void *)malloc(all_size);
    if (nd->data == NULL) {
      LOG(INFO) << "Failed to allocate CPU Memory";
      return kFailure;
    }

    if (fi->Read(nd->data, all_size) != all_size) {
      LOG(INFO) << "read param error.";
      return kFailure;
    }

  }

  // read nd names
  std::vector<std::string> keys;
  if (!(fi->Read(&keys))) {
    LOG(INFO) << "Invalid NDArray file format";
    return kFailure;
  }
  if (keys.size() != 0 && keys.size() != ndarrays.size()) {
    LOG(INFO) << "Invalid NDArray file format";
    return kFailure;
  }

  for (size_t i = 0; i < nd_size; ++i) {
    std::string name(keys[i].c_str() + 4);
    ndarrays[i]->name = name;
#if defined(DEBUG)
    LOG(INFO) << ndarrays[i]->name << ", shape: " << ndarrays[i]->shape;
#endif
  }

  delete fi;
  return kSuccess;
}

} // namespace parsend

#endif // PARSE_MX_ND_UTILS_HPP
