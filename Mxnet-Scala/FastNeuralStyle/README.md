# MXNET-Scala Human Activity Recognition
MXNet-scala module implementation of LSTM for Human Activity Recognition.

Based on: https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition

## Building

Tested on Ubuntu 14.04

###Requirements

* sbt 0.13
* Mxnet

###steps

1, compile Mxnet with CUDA, then compile the scala-pkg;

2, cd into Mxnet-Scala/HumanActivityRecognition, then mkdir lib;

3, copy your compiled mxnet-full_2.11-linux-x86_64-gpu-0.1.2-SNAPSHOT.jar into lib folder;

4, run sbt, compile the project

## Running

* cd scripts;
* bash run.sh

then have fun!
 

## Visualization of the training process

<img src="./visualize/result.png" width="800"/>



