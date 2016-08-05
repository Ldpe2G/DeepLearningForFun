package mxgan

import Ops.conv2DBnLeaky
import ml.dmlc.mxnet.Symbol

object Encoder {

  // Lenet before classification layer.
  def lenet(): Symbol = {
    val data = Symbol.Variable("data")
    // 28 x 28
    val conv1 = Symbol.Convolution("conv1")(Map("data" -> data, "kernel" -> "(5,5)", "num_filter" -> 20))
    val tanh1 = Symbol.Activation()(Map("data" -> conv1, "act_type" -> "tanh"))
    val pool1 = Symbol.Pooling()(Map("data" -> tanh1, "pool_type" -> "max", "kernel" -> "(2,2)", "stride" -> "(2,2)"))
    // second conv
    val conv2 = Symbol.Convolution("conv2")(Map("data" -> pool1, "kernel" -> "(5,5)", "num_filter" -> 50))
    val tanh2 = Symbol.Activation()(Map("data" -> conv2, "act_type" -> "tanh"))
    val pool2 = Symbol.Pooling()(Map("data" -> tanh2, "pool_type" -> "max", "kernel" -> "(2,2)", "stride" -> "(2,2)"))
    var d5 = Symbol.Flatten()(Map("data" -> pool2))
    d5 = Symbol.FullyConnected("fc1")(Map("data" -> d5, "num_hidden" -> 500))
    d5 = Symbol.Activation()(Map("data" -> d5, "act_type" -> "tanh"))
    d5    
  }

  // Conv net used in original DGCAN
  def dcgan(ngf: Int = 128): Symbol = {
    val data = Symbol.Variable("data")
    // 128, 16, 16
    var net = Symbol.Convolution("e1_conv")(Map("data" -> data, "kernel" -> "(4, 4)",
                                          "stride" -> "(2, 2)", "pad" -> "(1, 1)", "num_filter" -> ngf))
    net = Symbol.LeakyReLU("e1_act")(Map("data" -> net, "slope" -> 0.2f, "act_type" -> "leaky"))
    // 256, 8, 8
    net = conv2DBnLeaky(net, prefix = "e2", kernel = (4, 4), stride = (2, 2), pad = (1, 1), numFilter = ngf * 2)
    // 512, 4, 4
    net = conv2DBnLeaky(net,  prefix="e3", kernel = (4, 4), stride = (2, 2), pad = (1, 1), numFilter = ngf * 4)
    net = Symbol.Flatten()(Map("data" -> net))
    net
  }

}
