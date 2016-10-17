# MXNET-Scala Fast Neural Style
MXNet-scala module implementation of fast neural style[1].

Based on: 

https://github.com/jcjohnson/fast-neural-style and https://github.com/dmlc/mxnet/tree/master/example/neural-style/end_to_end

The results are not as good as the torch version but not bad anyway :)

Time: 0.075 seconds in average on NVIDIA GTX 1070 GPU for at resolution of 712 x 414.

## Results:

<div align='center'>
  <img src='datas/images/chicago.jpg' height="185px">
</div>
<br>
<div align='center'>
  <img src='datas/images/the_scream.jpg' height='174px'>
  <img src='datas/pretrain_models/scream/out.jpg' height="174px">
</div>
<br>
<div align='center'>
  <img src='datas/images/mosaic.jpg' height='174px'>
  <img src='datas/pretrain_models/mosaic/out.jpg' height="174px">
</div>
<br>
<div align='center'>
  <img src='datas/images/feathers.jpg' height='173px'>
  <img src='datas/pretrain_models/feathers/out.jpg' height="173px">
</div>
<br>
<div align='center'>
  <img src='datas/images/la_muse.jpg' height='173px'>
  <img src='datas/pretrain_models/la_muse/out.jpg' height="173px">
</div>
<br>
<div align='center'>
  <img src='datas/images/candy.jpg' height='174px'>
  <img src='datas/pretrain_models/candy/out.jpg' height="174px">
</div>
<br>
<div align='center'>
  <img src='datas/images/udnie.jpg' height='174px'>
  <img src='datas/pretrain_models/udnie/out.jpg' height="174px">
</div>


## Setup
Tested on Ubuntu 14.04

###Requirements

* [sbt 0.13]http://www.scala-sbt.org/
* [Mxnet]https://github.com/dmlc/mxnet

###steps

1, compile Mxnet with CUDA, then compile the [scala-pkg]https://github.com/dmlc/mxnet/tree/master/scala-package

2, under the Mxnet-Scala/FastNeuralStyle folder 
```bah
 mkdir lib;
```
3, copy your compiled mxnet-full_2.11-linux-x86_64-gpu-0.1.2-SNAPSHOT.jar into lib folder;

4, run `sbt` then compile the project

## Running on new images
The script `run_fast_neural_style.sh` in the scripts older, lets you use a pre-trained model to stylize new images:

```bash
java -Xmx1G -cp $CLASS_PATH \
	FastNeuralStyle \
	--model-path  $PREAREIN_MODEL \
	--input-image $INPUT_IMAGE \
	--output-path $OUTPUT_PATH \
	--gpu $GPU
```
You can run this script on CPU or GPU, 

for cpu set the GPU to -1;

for gpu plesase specifying the GPU on which to run.

## Training new models

```

### Pretrained Models
Download all pretrained style transfer models by running the script

```bash
bash models/download_style_transfer_models.sh
```

This will download ten model files (~200MB) to the folder `models/`.

You can [find instructions for training new models here](doc/training.md).

## License

Free for personal or research use.

## Reference
[1] Johnson, Justin, Alexandre Alahi, and Li Fei-Fei. "Perceptual losses for real-time style transfer and super-resolution." arXiv preprint arXiv:1603.08155.

