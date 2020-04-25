import mxnet as mx
import random
import numpy as np

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

data_shape = (1, 1, 3, 3)

model = mx.mod.Module(context=mx.cpu(), symbol=fc, data_names=['data'],
                      label_names=None)
model.bind(data_shapes=[('data', data_shape)],
           label_shapes=None,
           for_training=True)

model.init_params()

data_nd = mx.nd.array(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])).reshape(data_shape)
model.forward(mx.io.DataBatch([data_nd], []))

print('input: ')
print(data_nd)
print('conv weight: ')
print(model._exec_group.execs[0].arg_dict['conv_weight'])
print('conv bias: ')
print(model._exec_group.execs[0].arg_dict['conv_bias'])

print('fc weight: ')
print(model._exec_group.execs[0].arg_dict['fc_weight'])
print('fc bias: ')
print(model._exec_group.execs[0].arg_dict['fc_bias'])


print('python result: %s' % ', '.join([str(e) for e in model.get_outputs()[0].asnumpy()]))
print()


model.save_checkpoint('test', 0, save_optimizer_states=False)


