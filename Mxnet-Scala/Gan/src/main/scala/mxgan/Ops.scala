package mxgan

import org.apache.mxnet.Symbol
import org.apache.mxnet.Shape
import org.apache.mxnet.util.OptionConversion._

object Ops {

  val eps: Float = 1e-5f + 1e-12f

  // a deconv layer that enlarges the feature map
  def deconv2D(data: Symbol, iShape: Shape, oShape: Shape,
    kShape: (Int, Int), name: String, stride: (Int, Int) = (2, 2)): Symbol = {
    val targetShape = (oShape(oShape.length - 2), oShape(oShape.length - 1))
    val net = Symbol.api.Deconvolution(data,
                                       kernel = Shape(kShape._1, kShape._2),
                                       stride = Shape(stride._1, stride._2),
                                       target_shape = Shape(targetShape._1, targetShape._2),
                                       num_filter = oShape(0),
                                       no_bias = true)
    net
  }

  def deconv2DBnRelu(data: Symbol, prefix: String,
    iShape: Shape, oShape: Shape, kShape: (Int, Int)): Symbol = {
    var net = deconv2D(data, iShape, oShape, kShape, name = s"${prefix}_deconv")
    net = Symbol.api.BatchNorm(net, fix_gamma = true, eps = eps, name = s"${prefix}_bn")
    net = Symbol.api.relu(net, name = s"${prefix}_act")
    net
  }

  def deconv2DAct(data: Symbol, prefix: String, actType: String,
    iShape: Shape, oShape: Shape, kShape: (Int, Int)): Symbol = {
    var net = deconv2D(data, iShape, oShape, kShape, name = s"${prefix}_deconv")
    net = Symbol.api.Activation(net, act_type = actType, name = s"${prefix}_act")
    net
  }

  def conv2DBnLeaky(data: Symbol, prefix: String, kernel: (Int, Int),
    stride: (Int, Int), pad: (Int, Int), numFilter: Int): Symbol = {
    var net = Symbol.api.Convolution(data,
                                     kernel = Shape(kernel._1, kernel._2),
                                     stride = Shape(stride._1, stride._2),
                                     pad = Shape(pad._1, pad._2),
                                     num_filter = numFilter)
    net = Symbol.api.BatchNorm(net, fix_gamma = true, eps = eps, name = s"${prefix}_bn")
    net = Symbol.api.LeakyReLU(net, act_type = "leaky", name = s"${prefix}_leaky")
    net
  }

}
