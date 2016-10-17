# MXNET-Scala Fast Neural Style
MXNet-scala module implementation of fast neural style[1].

Based on: https://github.com/jcjohnson/fast-neural-style and https://github.com/dmlc/mxnet/tree/master/example/neural-style/end_to_end

The results are not as good as the torch version but not bad anyway :)

Time: 0.075 seconds in average on NVIDIA GTX 1070 GPU for at resolution of 712 x 414.

## Setup

First [install Torch](http://torch.ch/docs/getting-started.html#installing-torch), then
update / install the following packages:


<div align='center'>
  <img src='datas/images/candy.jpg' height='174px'>
  <img src='datas/pretrain_models/candy/out.jpg' height="174px">
  <img src='datas/pretrain_models/udnie/out.jpg' height="174px">
  <img src='datas/images/udnie.jpg' height='174px'>
  <br>
  <img src='datas/images/the_scream.jpg' height='174px'>
  <img src='datas/pretrain_models/scream/out.jpg' height="174px">
  <img src='datas/pretrain_models/mosaic/out.jpg' height="174px">
  <img src='datas/images/mosaic.jpg' height='174px'>
  <br>
  <img src='datas/images/feathers.jpg' height='173px'>
  <img src='datas/pretrain_models/feathers/out.jpg' height="173px">
  <img src='datas/pretrain_models/muse/out.jpg' height="173px">
  <img src='datas/images/la_muse.jpg' height='173px'>
</div>


## Running on new images
The script `fast_neural_style.lua` lets you use a trained model to stylize new images:

```bash
th fast_neural_style.lua \
  -model models/eccv16/starry_night.t7 \
  -input_image images/content/chicago.jpg \
  -output_image out.png
```

You can run the same model on an entire directory of images like this:

```bash
th fast_neural_style.lua \
  -model models/eccv16/starry_night.t7 \
  -input_dir images/content/ \
  -output_dir out/
```

You can control the size of the output images using the `-image_size` flag.

By default this script runs on CPU; to run on GPU, add the flag `-gpu`
specifying the GPU on which to run.

The full set of options for this script is [described here](doc/flags.md#fast_neural_stylelua).


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

