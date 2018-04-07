package mxgan

import org.apache.mxnet.Symbol
import org.apache.mxnet.Shape

object Ops {

  val eps: Float = 1e-5f + 1e-12f

  // a deconv layer that enlarges the feature map
  def deconv2D(data: Symbol, iShape: Shape, oShape: Shape,
    kShape: (Int, Int), name: String, stride: (Int, Int) = (2, 2)): Symbol = {
    val targetShape = (oShape(oShape.length - 2), oShape(oShape.length - 1))
    val net = Symbol.Deconvolution(name)()(Map(
                                           "data" -> data,
                                           "kernel" -> s"$kShape",
                                           "stride" -> s"$stride",
                                           "target_shape" -> s"$targetShape",
                                           "num_filter" -> oShape(0),
                                           "no_bias" -> true))
    net
  }

  def deconv2DBnRelu(data: Symbol, prefix: String,
    iShape: Shape, oShape: Shape, kShape: (Int, Int)): Symbol = {
    var net = deconv2D(data, iShape, oShape, kShape, name = s"${prefix}_deconv")
    net = Symbol.BatchNorm(s"${prefix}_bn")()(Map("data" -> net, "fix_gamma" -> true, "eps" -> eps))
    net = Symbol.Activation(s"${prefix}_act")()(Map("data" -> net, "act_type" -> "relu"))
    net
  }

  def deconv2DAct(data: Symbol, prefix: String, actType: String,
    iShape: Shape, oShape: Shape, kShape: (Int, Int)): Symbol = {
    var net = deconv2D(data, iShape, oShape, kShape, name = s"${prefix}_deconv")
    net = Symbol.Activation(s"${prefix}_act")()(Map("data" -> net, "act_type" -> actType))
    net
  }

  def conv2DBnLeaky(data: Symbol, prefix: String, kernel: (Int, Int),
    stride: (Int, Int), pad: (Int, Int), numFilter: Int): Symbol = {
    var net = Symbol.Convolution(s"${prefix}_conv")()(Map(
                                          "data" -> data,
                                          "kernel" -> s"$kernel",
                                          "stride" -> s"$stride",
                                          "pad" -> s"$pad",
                                          "num_filter" -> numFilter))
    net = Symbol.BatchNorm(s"${prefix}_bn")()(Map("data" -> net, "fix_gamma" -> true, "eps" -> eps))
    net = Symbol.LeakyReLU(s"${prefix}_leaky")()(Map("data" -> net, "act_type" -> "leaky"))
    net
  }

}
