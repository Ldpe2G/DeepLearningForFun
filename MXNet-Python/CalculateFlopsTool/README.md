# MXNET-Python Tool For Calculate Flops And Model Size

Implementation of the estimation of model size and flop counts for convolutional neural networks.

ref: https://github.com/albanie/convnet-burden

The estimation of flops only consider layers: Convolution, Deconvolution, FullyConnected, Pooling, relu

[Scala Version](https://github.com/Ldpe2G/DeepLearningForFun/tree/master/Mxnet-Scala/UsefulTools)

## Running

```bash
python calculateFlops.py -s symbols/caffenet-symbol.json -ds  data,1,3,224,224 -ls prob_label,1,1000

('flops: ', '723.007176', ' MFLOPS')
('model size: ', '232.563873291', ' MB')


python calculateFlops.py -s symbols/squeezenet_v1.0-symbol.json -ds  data,1,3,224,224 -ls prob_label,1,1000

('flops: ', '861.603864', ' MFLOPS')
('model size: ', '4.76235961914', ' MB')


python calculateFlops.py -s symbols/resnet-101-symbol.json -ds  data,1,3,224,224 -ls softmax_label,1,1000 

('flops: ', '7818.240488', ' MFLOPS')
('model size: ', '169.912773132', ' MB')

python calculateFlops.py -s symbols/resnext-101-64x4d-symbol.json -ds data,1,3,224,224 -ls softmax_label,1,1000

('flops: ', '15491.88196', ' MFLOPS')
('model size: ', '318.356620789', ' MB')


python calculateFlops.py -s symbols/fcn8s-symbol.json -ds  data,1,3,384,384 -ls softmax_label,1,21,384,384

('flops: ', '120420.573296', ' MFLOPS')
('model size: ', '513.037715912', ' MB')


python calculateFlops.py -s symbols/fcn32s-symbol.json -ds data,1,3,384,384 -ls softmax_label,1,21,384,384

('flops: ', '120265.786832', ' MFLOPS')
('model size: ', '519.382160187', ' MB')
```
