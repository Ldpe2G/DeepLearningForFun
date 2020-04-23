#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <map>
#include "mx_nd_utils.hpp"

int main(int argc, char** argv) {

  std::string param_file = argc > 1 ? argv[1] : "../testData/test-0000.params";

  std::vector<parsend::NDArray *> ndarrays; // remember to release memeory

  // load parameters
  int32_t r = parsend::loadNDArray(ndarrays, param_file);
  if (r == parsend::kSuccess) {
    std::cout <<  "read nd file successfully." <<std::endl;
  }

  std::map<std::string, parsend::NDArray *> map; 
  // print parameter name and shape
  for (auto nd_p : ndarrays) {
    map[nd_p->name] = nd_p;
    std::cout << "weight name: " << nd_p->name << ", shape: " << nd_p->shape << std::endl;
  }


  // perform the forward operation 
  std::vector<float> input = {1, 2, 3, 4};

  std::vector<float> result;
  result.resize(101);

  float *weight = static_cast<float* >(map["fc1_weight"]->data);
  float *bias = static_cast<float* >(map["fc1_bias"]->data);
  for (int i=0; i<100; ++i) {
    float sum = 0;
    for (int j=0; j<4; ++j) {
      sum += weight[i * 4 + j] * input[j];
    }
    result[i] = sum + bias[i];
  }

  weight = static_cast<float* >(map["fc2_weight"]->data);
  bias = static_cast<float* >(map["fc2_bias"]->data);
  result[100] = 0;
  for (int j=0; j<100; ++j) {
    result[100] += weight[j] * result[j];
  }
  result[100] += bias[0];

  std::cout << "result: " << result[100] << std::endl;

  for (int i = 0; i < ndarrays.size(); ++i) {
    delete ndarrays[i];
  }

  return 0;
}

