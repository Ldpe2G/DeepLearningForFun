# MXNET-Scala Playing Flappy Bird Using Deep Reinforcement Learning
MXNet-scala module implementation of DQN to Play Flappy Bird.

Based on: https://github.com/li-haoran/DRL-FlappyBird

##result:
<img src="./results/optimised.gif"/>

## Setup
Tested on Ubuntu 14.04

###Requirements

* sbt 0.13 http://www.scala-sbt.org/
* Mxnet https://github.com/dmlc/mxnet

###steps

1, compile Mxnet with CUDA, then compile the scala-pkg，doc： https://github.com/dmlc/mxnet/tree/master/scala-package

2, under the Mxnet-Scala/DRLFlappyBird folder 
```bah
 mkdir lib;
```
3, copy your compiled mxnet-full_2.11-linux-x86_64-gpu-0.1.2-SNAPSHOT.jar into lib folder;

4, run `sbt` then compile the project


## Training
using the script `scripts/run.sh`:

```bash
#### training #####

java -Xmx4G -cp $CLASS_PATH \
	FlappyBirdDQN \
	--gpu $GPU \
	--save-model-path $SAVE_MODRL_PATH \
	--resources-path $RESOURCES_PATH
```

## Running with pretrain models
using the script `scripts/run.sh`, comment the training part and uncomment the folllowing line:

```bash
#### resume training ####

RESUME_MODRL_PATH=$ROOT/models/pretrain-model/network-dqn_mx46000.params

java -Xmx4G -cp $CLASS_PATH \
   FlappyBirdDQN \
  	--gpu $GPU \
  	--save-model-path $SAVE_MODRL_PATH \
  	--resources-path $RESOURCES_PATH \
  	--resume-model-path $RESUME_MODRL_PATH
```

Have fun !!
