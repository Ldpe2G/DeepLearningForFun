# MXNET-Scala Self-Norm Nets
MXNet-scala module implementation of Self-normalizing networks[1].

Based on: https://github.com/bioinf-jku/SNNs

## Building

Tested on Ubuntu 14.04

### Requirements

* sbt 0.13
* Mxnet

### steps

1, compile Mxnet with CUDA, then compile the scala-pkg;

2, 
```bash
cd Mxnet-Scala/SelfNormNets
mkdir lib
```

3, copy your compiled mxnet-full_2.11-linux-x86_64-gpu-0.10.1-SNAPSHOT.jar into lib folder;

4, run sbt, compile the project

## Running

1, use datas/get_mnist_data.sh script to download the mnist dataset

2, run `trainSNN_CNN_MNIST.sh` or `trainSNN_MLP_MNIST.sh` under scripts folder

## Training logs

### Train MLP

#### logs
```
bash trainSNN_MLP_MNIST.sh

Epoch[0] Train-accuracy=0.86191666
Epoch[0] Time cost=1646
Epoch[0] Validation-accuracy=0.9341
Epoch[1] Train-accuracy=0.9213667
Epoch[1] Time cost=1478
Epoch[1] Validation-accuracy=0.9485
Epoch[2] Train-accuracy=0.9421667
Epoch[2] Time cost=1428
Epoch[2] Validation-accuracy=0.9402
Epoch[3] Train-accuracy=0.9501
Epoch[3] Time cost=1415
Epoch[3] Validation-accuracy=0.9669
Epoch[4] Train-accuracy=0.9571667
Epoch[4] Time cost=1604
Epoch[4] Validation-accuracy=0.9623
Epoch[5] Train-accuracy=0.96195
Epoch[5] Time cost=1457
Epoch[5] Validation-accuracy=0.9614
Epoch[6] Train-accuracy=0.9679667
Epoch[6] Time cost=1591
Epoch[6] Validation-accuracy=0.9673
Epoch[7] Train-accuracy=0.97048336
Epoch[7] Time cost=1629
Epoch[7] Validation-accuracy=0.9639
Epoch[8] Train-accuracy=0.9719333
Epoch[8] Time cost=1668
Epoch[8] Validation-accuracy=0.9703
Epoch[9] Train-accuracy=0.9753
Epoch[9] Time cost=1662
Epoch[9] Validation-accuracy=0.9728
Epoch[10] Train-accuracy=0.9769
Epoch[10] Time cost=1526
Epoch[10] Validation-accuracy=0.9752
Epoch[11] Train-accuracy=0.9784333
Epoch[11] Time cost=1487
Epoch[11] Validation-accuracy=0.9709
Epoch[12] Train-accuracy=0.98066664
Epoch[12] Time cost=1609
Epoch[12] Validation-accuracy=0.9753
Epoch[13] Train-accuracy=0.98113334
Epoch[13] Time cost=1475
Epoch[13] Validation-accuracy=0.9725
Epoch[14] Train-accuracy=0.98215
Epoch[14] Time cost=1477
Epoch[14] Validation-accuracy=0.9749
```

### Compare selu with relu 
#### logs
```bash
bash trainSNN_CNN_MNIST.sh

Epoch[0] SNN Train-accuracy=0.88266224
Epoch[0] ReLU Train-accuracy=0.807926
Epoch[1] SNN Train-accuracy=0.9415899
Epoch[1] ReLU Train-accuracy=0.8241854
Epoch[2] SNN Train-accuracy=0.95097154
Epoch[2] ReLU Train-accuracy=0.8243189
Epoch[3] SNN Train-accuracy=0.95880073
Epoch[3] ReLU Train-accuracy=0.833734
Epoch[4] SNN Train-accuracy=0.9629741
Epoch[4] ReLU Train-accuracy=0.82568777
Epoch[5] SNN Train-accuracy=0.96793205
Epoch[5] ReLU Train-accuracy=0.8318643
Epoch[6] SNN Train-accuracy=0.9703693
Epoch[6] ReLU Train-accuracy=0.8342181
Epoch[7] SNN Train-accuracy=0.97163796
Epoch[7] ReLU Train-accuracy=0.83628803
Epoch[8] SNN Train-accuracy=0.9741086
Epoch[8] ReLU Train-accuracy=0.8316807
Epoch[9] SNN Train-accuracy=0.9753105
Epoch[9] ReLU Train-accuracy=0.8397269
SNN Validation-accuracy=0.96334136
ReLU Validation-accuracy=0.9423077
```

## Referneces
[1] Klambauer, Günter, et al. "Self-Normalizing Neural Networks." arXiv preprint arXiv:1706.02515 (2017).



