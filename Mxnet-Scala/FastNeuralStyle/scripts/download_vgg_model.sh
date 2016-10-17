#!/bin/bash

ROOT=$(cd "$(dirname $0)/.."; pwd)

mkdir -p $ROOT/datas/vggmodel

cd $ROOT/datas/vggmodel
wget https://github.com/dmlc/web-data/raw/master/mxnet/neural-style/model/vgg19.params
cd -

