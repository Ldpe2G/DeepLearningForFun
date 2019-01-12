# MXNET-Scala TrainQuantization
Simply implementation of Quantization Aware Training[1] with MXNet-scala module. 


## Setup
Tested on Ubuntu 14.04

### Requirements

* sbt 0.13 http://www.scala-sbt.org/
* Mxnet v1.4 https://github.com/dmlc/mxnet

### steps

1, compile MXNet with CUDA, then compile the scala-pkg，doc： https://github.com/dmlc/mxnet/tree/master/scala-package

2, in the Mxnet-Scala/TrainQuantization folder 
```bash
 mkdir lib;
```
3, copy the compiled mxnet-full_2.11-linux-x86_64-gpu-1.5.0-SNAPSHOT.jar into lib folder;

4, run `sbt` and then compile the project

## Train vgg on Cifar10
Using the script `train_vgg16_cifar10.sh` under the scripts folder to train vgg net from scratch on Cifar10:

```bash
FINETUNE_MODEL_EPOCH=-1
FINETUNE_MODEL_PREFIX=$ROOT/models/
```
Or you can finetune with the provided pretrained model:

```bash
FINETUNE_MODEL_EPOCH=46
FINETUNE_MODEL_PREFIX=$ROOT/models/cifar10_vgg16_acc_0.8772035
```

I did not use any data augmentation and tune the hyper-parameters during training, the best accuracy I got was 0.877, worse than the best accracy 0.93 reported on Cifar10.

## Train vgg with fake quantization on Cifar10
Using the script `train_quantize_vgg16_cifar10.sh` under the scripts folder to train vgg with fake quantization on Cifar10,
you must provide the pretrained model:

```bash
FINETUNE_MODEL_EPOCH=46
FINETUNE_MODEL_PREFIX=$ROOT/models/cifar10_vgg16_acc_0.8772035
```

If everything goes right, you can get almost the same accuray with pretrained model after serveral epoch.

## Test vgg with simulated quantization on Cifar10
Using the script `test_quantize_vgg16_cifar10.sh` under the scripts folder to test pretrained fake quantization vgg with simulated quantization on Cifar10, you must provide the pretrained model:

```bash
FINETUNE_MODEL_EPOCH=57
FINETUNE_MODEL_PREFIX=$ROOT/models/cifar10_quantize_vgg16_acc_0.877504
```

## Warning
Currently there is memory leak problems some where in the code, but I can't figure out the reason. You will see the memory usage keep increasing when you run the tranining script. So remenber to stop the traning script when memory usage is too high, and you can resume the training process with saved model previously.

## Reference
[1] Quantizing deep convolutional networks for efficient inference: A whitepaper. https://arxiv.org/pdf/1806.08342.pdf
