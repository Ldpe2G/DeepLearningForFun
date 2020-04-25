# Parsing MXNet Parameter File In C++

This small project aims to extract parameters from pretrained MXNet `.params` file in C++.

If you want to run your MXNet pretrained model on mobile devices with other light weight frameworks, you need to first extract the pretrained parameters from the `.params` file.

This project shows how to parse the `.params` file and extract the weights.

## Build The Project

### Compile The Project

```bash
bash build_run.sh
```

## Testing

The `test.py` script constructs a simple nerwork with only one convolution and one fc layer:

```python
import mxnet as mx

data = mx.sym.Variable('data')

conv_prefix = 'conv'
conv_weight = mx.sym.Variable(conv_prefix + '_weight', init=mx.init.Normal())
conv_bias = mx.sym.Variable(conv_prefix + '_bias', init=mx.init.Normal())
fc = mx.sym.Convolution(data=data, weight=conv_weight, bias=conv_bias,
                        kernel=(3,3), pad=(1,1), stride=(1,1), 
                        num_filter=1, no_bias=False, name=conv_prefix)

fc_prefix = 'fc'
fc_weight = mx.sym.Variable(fc_prefix + '_weight', init=mx.init.Normal())
fc_bias = mx.sym.Variable(fc_prefix + '_bias', init=mx.init.Normal())
fc = mx.sym.FullyConnected(data=fc, weight=fc_weight, bias=fc_bias, num_hidden=1, no_bias=False, name='fc')

```

And it will generate a `.params` file contains random init weights by running the script.

```bash
cd testData && python test.py
```

Then you can load the `.params` file and perform forward operation in C++.

```bash
cd build && ./read_nd ../testData/test-0000.params
```
And you will see the same forward result as the output of python script.


## Some Limitations & Future Works
1. I have only implemented the parsing of normal storage type NDArray, if your `.params` file contains sparse NDArray, this tool will fail to parse the file.
