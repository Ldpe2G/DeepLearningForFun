#include <iostream>
#include <map>
#include "mx_nd_utils.hpp"

int main(int argc, char** argv) {

  std::string param_file = argc > 1 ? argv[1] : "../testData/test-0000.params";

  std::vector<parsend::NDArray *> ndarrays; // remember to release memeory

  // load parameters
  int32_t r = parsend::loadNDArray(ndarrays, param_file);
  if (r == parsend::kSuccess) {
    std::cout <<  "C++ read nd file successfully." <<std::endl;
  }

  std::map<std::string, parsend::NDArray *> map; 
  // print parameter name and shape
  for (auto nd_p : ndarrays) {
    map[nd_p->name] = nd_p;
    std::cout << "weight name: " << nd_p->name << std::endl;

    std::cout << "shape: \n";
    int size = 1;
    for (int i = 0; i < nd_p->shape.size() - 1; ++i) {
      std::cout << nd_p->shape[i] << ", ";
      size *= nd_p->shape[i];
    }
    size *= nd_p->shape[nd_p->shape.size() - 1];
    std::cout << nd_p->shape[nd_p->shape.size() - 1] << std::endl;

    // const float *ptr = static_cast<const float *>(nd_p->data);
    // for (int i = 0; i < size; ++i) {
    //   std::cout << ptr[i] << ", ";
    // }
    // std::cout << std::endl;
  }

  // perform the forward operation 
  std::vector<float> input = {1, 2, 3, 4, 5, 6, 7, 8, 9};

  std::vector<float> results;
  results.resize(10);

  // perform naive convolution computation
  parsend::NDArray *conv_weight_nd = map["conv_weight"] ;
  const float *weight = static_cast<const float* >(conv_weight_nd->data);
  const float *bias = static_cast<const float* >(map["conv_bias"]->data);

  const int Kh = 3;
  const int Kw = 3;
  const int Hout = (3 + 2 - Kh) + 1;
  const int Wout = (3 + 2 - Kw) + 1;

  const int halfKh = Kh / 2;
  const int halfKw = Kw / 2;

  int idx = 0;
  for (int h = 0; h < Hout; ++h) {
    int start_h = std::max(0, h - halfKh);
    int end_h = std::min(Hout - 1, h + halfKh);
    for (int w = 0; w < Wout; ++w) {
      int start_w = std::max(0, w - halfKw);
      int end_w = std::min(Wout - 1, w + halfKw);
      float result = bias[0];
      for (int sh = start_h; sh <= end_h; ++sh) {
        for (int sw = start_w; sw <= end_w; ++sw) {
          result += input[sh * 3 + sw] * weight[(sh - h + halfKh) * Kw + (sw - w + halfKw)];
        }
      }
      results[idx ++] = result;
    }
  }

  // perform fullyconnect computation
  parsend::NDArray *fc_weight_nd = map["fc_weight"] ;
  weight = static_cast<float* >(fc_weight_nd->data);
  bias = static_cast<float* >(map["fc_bias"]->data);


  for (int i = 0; i < fc_weight_nd->shape[0]; ++i) {
    float sum = 0;
    for (int j = 0; j < fc_weight_nd->shape[1]; ++j) {
      sum += weight[i * fc_weight_nd->shape[1] + j] * results[j];
    }
    results[i+9] = sum + bias[i];
  }

  std::cout << "C++ result: " << results[9] << std::endl;

  for (int i = 0; i < ndarrays.size(); ++i) {
    delete ndarrays[i];
  }

  return 0;
}

