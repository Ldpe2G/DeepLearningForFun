import mxnet as mx
import random
import numpy as np

data = mx.sym.Variable('data')
fc = mx.sym.FullyConnected(data=data, num_hidden=100, no_bias=False, name='fc1')
fc = mx.sym.FullyConnected(data=fc, num_hidden=1, no_bias=False, name='fc2')


data_shape = (1, 4)

model = mx.mod.Module(context=mx.cpu(), symbol=fc, data_names=['data'],
                      label_names=None)
model.bind(data_shapes=[('data', data_shape)],
           label_shapes=None,
           for_training=True)

model.init_params()

data_nd = mx.nd.array(np.array([[1,2,3,4]]))
model.forward(mx.io.DataBatch([data_nd], []))

print('input: %s' % ', '.join([str(e) for e in data_nd.asnumpy()]))
print('result: %s' % ', '.join([str(e) for e in model.get_outputs()[0].asnumpy()]))

model.save_checkpoint('test', 0, save_optimizer_states=False)


