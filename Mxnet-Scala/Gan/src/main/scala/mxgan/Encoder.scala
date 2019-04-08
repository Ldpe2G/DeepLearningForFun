package mxgan

import Ops.conv2DBnLeaky
import org.apache.mxnet._
import org.apache.mxnet.util.OptionConversion._

object Encoder {

  // Lenet before classification layer.
  def lenet(): Symbol = {
    val data = Symbol.Variable("data")
    // 28 x 28
    val conv1 = Symbol.api.Convolution(data, kernel = Shape(5,5), num_filter = 20, name = "conv1")
    val tanh1 = Symbol.api.tanh(conv1)
    val pool1 = Symbol.api.Pooling(tanh1, pool_type = "max", kernel = Shape(2,2), stride = Shape(2,2))
    // second conv
    val conv2 = Symbol.api.Convolution(pool1, kernel = Shape(5,5), num_filter = 50, name = "conv2")
    val tanh2 = Symbol.api.tanh(conv2)
    val pool2 = Symbol.api.Pooling(tanh2, pool_type = "max", kernel = Shape(2,2), stride = Shape(2,2))
    var d5 = Symbol.api.Flatten(pool2)
    d5 = Symbol.api.FullyConnected(d5, num_hidden = 500, name = "fc1")
    d5 = Symbol.api.tanh(d5)
    d5    
  }

  // Conv net used in original DGCAN
  def dcgan(ngf: Int = 128): Symbol = {
    val data = Symbol.Variable("data")
    // 128, 16, 16
    var net = Symbol.api.Convolution(data, kernel = Shape(4, 4),
                                     stride = Shape(2, 2), pad = Shape(1, 1), num_filter = ngf, name = "e1_conv")
    net = Symbol.api.LeakyReLU(net, slope = 0.2f, act_type = "leaky", name = "e1_act")
    // 256, 8, 8
    net = conv2DBnLeaky(net, prefix = "e2", kernel = (4, 4), stride = (2, 2), pad = (1, 1), numFilter = ngf * 2)
    // 512, 4, 4
    net = conv2DBnLeaky(net,  prefix="e3", kernel = (4, 4), stride = (2, 2), pad = (1, 1), numFilter = ngf * 4)
    net = Symbol.api.Flatten(net)
    net
  }

}
