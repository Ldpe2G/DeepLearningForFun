# MXNET-Scala CycleGAN
MXNet-scala module implementation of cycleGAN[1].

Based on: 
https://github.com/junyanz/CycleGAN

So far, I have tried all the tasks and done a lot of experiments., but just success in two tasks: "apple2orange" and "photo2ukiuoe".

I think I have followed the torch implementation completely and couldn't locate the problem :( .


## Results:

<div align='center'>
  <img src='results/apple2orange.png' height='150px'>
</div>

<div align='center'>
  <img src='results/orange2apple.png' height='150px'>
</div>

<div align='center'>
  <img src='results/photo2ukiuoe.png' height='150px'>
</div>


### Requirements

* sbt 0.13 http://www.scala-sbt.org/
* Mxnet https://github.com/dmlc/mxnet

### steps

1, compile Mxnet with CUDA, then compile the scala-pkg，doc： https://github.com/dmlc/mxnet/tree/master/scala-package

2, under the Mxnet-Scala/FastNeuralStyle folder 
```bah
 mkdir lib;
```
3, copy your compiled mxnet-full_2.11-linux-x86_64-gpu-0.1.2-SNAPSHOT.jar into lib folder;

4, run `sbt` then compile the project

## Testing


## Training new models

## License
MIT

## Reference
[1] Zhu, Jun Yan, et al. "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks." 2017.
