#ifndef PARSE_MX_ND_UTILS_HPP
#define PARSE_MX_ND_UTILS_HPP

#include <vector>
#include <iostream>
#include <fstream>
#include <map>
#include <string>

namespace parsend {
// Lowest NDArray object
// Can be regard as simple interpreter of void* 
// point.
class NDArray {
public:
  NDArray() {}
  ~NDArray() {
    if (data != NULL) free(data);
  }

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
    case kCSRStorage: std::cout << "Not supported storage type" << stype; break; //num = 2; break;
    case kRowSparseStorage: std::cout << "Not supported storage type" << stype; break; //num = 1; break;
     default: std::cout << "Unknown storage type" << stype; break;
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
      std::cout << "Unknown data type";
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
    std::cout << "Unknown type enum " << type;     \
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

static bool Read(std::FILE *fp, void *ptr, size_t size) {
  return std::fread(ptr, 1, size, fp) == size;
} 

int32_t loadNDArrayV2(std::vector<NDArray *>& ndarrays, std::string param_file) {

  std::FILE *fp = fopen(param_file.c_str(), "rb");

  uint64_t header, reserved;
  if (!Read(fp, (void*)(&header), sizeof(uint64_t))) {
    std::cout << "Invalid NDArray file format";
    return kFailure;
  }

  if (!Read(fp, (void*)(&reserved), sizeof(uint64_t))) {
    std::cout << "Invalid NDArray file format";
    return kFailure;
  }

  if (header != kMXAPINDArrayListMagic) {
    std::cout << "Invalid NDArray file format";
    return kFailure;
  }

  uint64_t nd_size;
  if (!Read(fp, (void*)(&nd_size), sizeof(uint64_t))) {
    std::cout << "read param error.";
    return kFailure;
  }

  size_t size = static_cast<size_t>(nd_size);
  
  ndarrays.resize(size);

  // read nd data
  for (size_t i = 0; i < nd_size; ++i) {
    NDArray* nd = new NDArray;
    ndarrays[i] = nd;

    uint32_t magic;
    if (!Read(fp, (void*)(&magic), sizeof(uint32_t))) {
      std::cout << "read param error.";
      return kFailure;
    }

    if (magic != NDARRAY_V2_MAGIC) {
      std::cout << "parsing ndarray not V2 format, not support yet!!";
      return kFailure;
    }

    // load storage type
    int32_t stype;
    if (!Read(fp, (void*)(&stype), sizeof(int32_t))) {
      std::cout << "read param error.";
      return kFailure;
    }

    const int32_t nad = num_aux_data(static_cast<NDArrayStorageType>(stype));

    // load shape
    uint32_t ndim_{0};
    if (!Read(fp, (void*)(&ndim_), sizeof(uint32_t))) {
      std::cout << "read param error.";
      return kFailure;
    }

    size_t nread = sizeof(int64_t) * ndim_;
    int64_t *data_heap_ = new int64_t[ndim_];
    if (!Read(fp, (void*)data_heap_, nread)) {
      std::cout << "read param error.";
      return kFailure;
    }

    int64_t size = 1;
    for (uint32_t i=0; i<ndim_;++i) {
      size *= data_heap_[i];
      nd->shape.push_back(data_heap_[i]);
    }

    delete[] data_heap_;

    // load context 
    DeviceType dev_type;
    int32_t dev_id;
    if (!Read(fp, (void*)(&dev_type), sizeof(dev_type))) {
      std::cout << "read param error.";
      return kFailure;
    }

    if (!Read(fp, (void*)(&dev_id), sizeof(int32_t))) {
      std::cout << "read param error.";
      return kFailure;
    }

    // load type flag
    int32_t type_flag;
    if (!Read(fp, (void*)(&type_flag), sizeof(int32_t))) {
      std::cout << "read param error.";
      return kFailure;
    }

    nd->type = type2str(type_flag);

    size_t all_size = size * mshadow_sizeof(type_flag);

    nd->numOfBytes = all_size;
    // int ret = posix_memalign(&(nd->data), alignment_, all_size);
    nd->data = (void *)malloc(all_size);
    if (nd->data == NULL) {
      std::cout << "Failed to allocate CPU Memory";
      return kFailure;
    }

    if (!Read(fp, nd->data, nd->numOfBytes)) {
      std::cout << "read param error.";
      return kFailure;
    }

  }

  // read nd names
  std::vector<std::string> keys;
  uint64_t keysLen;
  if (!Read(fp, (void*)(&keysLen), sizeof(uint64_t))) {
    std::cout << "read param error.";
    return kFailure;
  }
  keys.resize(keysLen);

  for (uint64_t k = 0; k < keysLen; ++k) {
     uint64_t stringLen;
     if (!Read(fp, (void*)(&stringLen), sizeof(uint64_t))) {
      std::cout << "read param error.";
      return kFailure;
    }
    size_t size = static_cast<size_t>(stringLen);
    keys[k].resize(size);
    if (size != 0) {
      size_t nbytes = sizeof(char) * size;
      if (!Read(fp, (void*)(&(keys[k][0])), nbytes)) {
        std::cout << "read param error.";
        return kFailure;
      }
    }
  }

  if (keys.size() != 0 && keys.size() != ndarrays.size()) {
    std::cout << "Invalid NDArray file format";
    return kFailure;
  }

  for (size_t i = 0; i < nd_size; ++i) {
    std::string name(keys[i].c_str() + 4);
    ndarrays[i]->name = name;
  }

  std::fclose(fp);
  return kSuccess;
}

} // namespace parsend

#endif // PARSE_MX_ND_UTILS_HPP
