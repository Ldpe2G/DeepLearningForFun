# MXNET-Scala Center Loss
MXNet-scala module implementation of Center Loss[1].

Based on: https://github.com/pangyupo/mxnet_center_loss

## Building

Tested on Ubuntu 14.04

### Requirements

* sbt 0.13
* Mxnet

### steps

1, compile Mxnet with CUDA, then compile the scala-pkg;

2, 
```bash
cd Mxnet-Scala/CenterLoss
mkdir lib
```

3, copy your compiled mxnet-full_2.11-linux-x86_64-gpu-0.1.2-SNAPSHOT.jar into lib folder;

4, run sbt, compile the project

## Running

1, use datas/get_mnist_data.sh script to download the mnist dataset

2, 
```bash
cd scripts
bash run_center_loss.sh
```

then have fun!

## Training logs

### Train with center loss

set WITH_CENTER_LOSS flag to 1 in run_center_loss.sh
```bash
WITH_CENTER_LOSS=1
```

#### logs
```
Epoch[0] Train-accuracy=(accuracy,0.85013336)
Epoch[0] Time cost=3207
Epoch[0] Validation-accuracy=(accuracy,0.9216)
Epoch[1] Train-accuracy=(accuracy,0.9300333)
Epoch[1] Time cost=2622
Epoch[1] Validation-accuracy=(accuracy,0.9403)
Epoch[2] Train-accuracy=(accuracy,0.9473)
Epoch[2] Time cost=2624
Epoch[2] Validation-accuracy=(accuracy,0.9526)
Epoch[3] Train-accuracy=(accuracy,0.95841664)
Epoch[3] Time cost=2576
Epoch[3] Validation-accuracy=(accuracy,0.9589)
Epoch[4] Train-accuracy=(accuracy,0.9662)
Epoch[4] Time cost=2579
Epoch[4] Validation-accuracy=(accuracy,0.9627)
Epoch[5] Train-accuracy=(accuracy,0.97108334)
Epoch[5] Time cost=2573
Epoch[5] Validation-accuracy=(accuracy,0.9659)
Epoch[6] Train-accuracy=(accuracy,0.97475)
Epoch[6] Time cost=2678
Epoch[6] Validation-accuracy=(accuracy,0.9691)
Epoch[7] Train-accuracy=(accuracy,0.9777667)
Epoch[7] Time cost=2599
Epoch[7] Validation-accuracy=(accuracy,0.9715)
Epoch[8] Train-accuracy=(accuracy,0.98043334)
Epoch[8] Time cost=2591
Epoch[8] Validation-accuracy=(accuracy,0.972)
Epoch[9] Train-accuracy=(accuracy,0.98275)
Epoch[9] Time cost=2573
Epoch[9] Validation-accuracy=(accuracy,0.9731)
Epoch[10] Train-accuracy=(accuracy,0.9848167)
Epoch[10] Time cost=2546
Epoch[10] Validation-accuracy=(accuracy,0.9735)
Epoch[11] Train-accuracy=(accuracy,0.98663336)
Epoch[11] Time cost=2721
Epoch[11] Validation-accuracy=(accuracy,0.9743)
Epoch[12] Train-accuracy=(accuracy,0.98836666)
Epoch[12] Time cost=2884
Epoch[12] Validation-accuracy=(accuracy,0.9744)
Epoch[13] Train-accuracy=(accuracy,0.9895167)
Epoch[13] Time cost=2728
Epoch[13] Validation-accuracy=(accuracy,0.975)
Epoch[14] Train-accuracy=(accuracy,0.991)
Epoch[14] Time cost=3003
Epoch[14] Validation-accuracy=(accuracy,0.9748)
Epoch[15] Train-accuracy=(accuracy,0.99215)
Epoch[15] Time cost=3490
Epoch[15] Validation-accuracy=(accuracy,0.9757)
Epoch[16] Train-accuracy=(accuracy,0.9933)
Epoch[16] Time cost=2893
Epoch[16] Validation-accuracy=(accuracy,0.9761)
Epoch[17] Train-accuracy=(accuracy,0.9942)
Epoch[17] Time cost=3126
Epoch[17] Validation-accuracy=(accuracy,0.9763)
Epoch[18] Train-accuracy=(accuracy,0.99515)
Epoch[18] Time cost=2716
Epoch[18] Validation-accuracy=(accuracy,0.9769)
Epoch[19] Train-accuracy=(accuracy,0.9960333)
Epoch[19] Time cost=2808
Epoch[19] Validation-accuracy=(accuracy,0.9773)
```

### Train without center loss

set WITH_CENTER_LOSS flag to 0 in run_center_loss.sh
```bash
WITH_CENTER_LOSS=0
```

#### logs
```bash
Epoch[0] Train-accuracy=(accuracy,0.8515667)
Epoch[0] Time cost=740
Epoch[0] Validation-accuracy=(accuracy,0.9209)
Epoch[1] Train-accuracy=(accuracy,0.93093336)
Epoch[1] Time cost=319
Epoch[1] Validation-accuracy=(accuracy,0.9428)
Epoch[2] Train-accuracy=(accuracy,0.94818336)
Epoch[2] Time cost=290
Epoch[2] Validation-accuracy=(accuracy,0.952)
Epoch[3] Train-accuracy=(accuracy,0.95853335)
Epoch[3] Time cost=260
Epoch[3] Validation-accuracy=(accuracy,0.957)
Epoch[4] Train-accuracy=(accuracy,0.96555)
Epoch[4] Time cost=270
Epoch[4] Validation-accuracy=(accuracy,0.9593)
Epoch[5] Train-accuracy=(accuracy,0.97003335)
Epoch[5] Time cost=261
Epoch[5] Validation-accuracy=(accuracy,0.9623)
Epoch[6] Train-accuracy=(accuracy,0.9743)
Epoch[6] Time cost=265
Epoch[6] Validation-accuracy=(accuracy,0.9655)
Epoch[7] Train-accuracy=(accuracy,0.9774167)
Epoch[7] Time cost=263
Epoch[7] Validation-accuracy=(accuracy,0.9682)
Epoch[8] Train-accuracy=(accuracy,0.9797)
Epoch[8] Time cost=284
Epoch[8] Validation-accuracy=(accuracy,0.9697)
Epoch[9] Train-accuracy=(accuracy,0.9820167)
Epoch[9] Time cost=267
Epoch[9] Validation-accuracy=(accuracy,0.971)
Epoch[10] Train-accuracy=(accuracy,0.98391664)
Epoch[10] Time cost=249
Epoch[10] Validation-accuracy=(accuracy,0.9713)
Epoch[11] Train-accuracy=(accuracy,0.98578334)
Epoch[11] Time cost=263
Epoch[11] Validation-accuracy=(accuracy,0.9724)
Epoch[12] Train-accuracy=(accuracy,0.9870833)
Epoch[12] Time cost=247
Epoch[12] Validation-accuracy=(accuracy,0.9732)
Epoch[13] Train-accuracy=(accuracy,0.9885333)
Epoch[13] Time cost=259
Epoch[13] Validation-accuracy=(accuracy,0.9728)
Epoch[14] Train-accuracy=(accuracy,0.98975)
Epoch[14] Time cost=249
Epoch[14] Validation-accuracy=(accuracy,0.9731)
Epoch[15] Train-accuracy=(accuracy,0.9910333)
Epoch[15] Time cost=254
Epoch[15] Validation-accuracy=(accuracy,0.9733)
Epoch[16] Train-accuracy=(accuracy,0.99196666)
Epoch[16] Time cost=252
Epoch[16] Validation-accuracy=(accuracy,0.9735)
Epoch[17] Train-accuracy=(accuracy,0.9931)
Epoch[17] Time cost=252
Epoch[17] Validation-accuracy=(accuracy,0.974)
Epoch[18] Train-accuracy=(accuracy,0.9939167)
Epoch[18] Time cost=271
Epoch[18] Validation-accuracy=(accuracy,0.9741)
Epoch[19] Train-accuracy=(accuracy,0.99478334)
Epoch[19] Time cost=250
Epoch[19] Validation-accuracy=(accuracy,0.9745)
```

## Referneces
[1] Wen, Yandong, et al. "A discriminative feature learning approach for deep face recognition." European Conference on Computer Vision. Springer International Publishing, 2016.



