package mxgan

import ml.dmlc.mxnet.Symbol
import ml.dmlc.mxnet.Shape
import Ops._

object Generator {

  // DCGAN that generates 28x28 images.
  def dcgan28x28(oShape: Shape, finalAct: String, ngf: Int = 128): Symbol = {
    assert(oShape(oShape.length - 1) == 28)
    assert(oShape(oShape.length - 2) == 28)
    val code = Symbol.Variable("code")
    var net = Symbol.FullyConnected("g1")(Map("data" -> code, "num_hidden" -> 4 * 4 * ngf * 4, "no_bias" -> true))
    net = Symbol.Activation("gact1")(Map("data" -> net, "act_type" -> "relu"))
    // 4 x 4
    net = Symbol.Reshape()(Map("data" -> net, "shape" -> s"(-1, ${ngf * 4}, 4, 4)"))
    // 8 x 8
    net = deconv2DBnRelu(net, prefix = "g2", iShape = Shape(ngf * 4, 4, 4), oShape = Shape(ngf * 2, 8, 8), kShape = (3, 3))
    // 14x14
    net = deconv2DBnRelu(net, prefix = "g3", iShape = Shape(ngf * 2, 8, 8), oShape = Shape(ngf, 14, 14), kShape = (4, 4))
    // 28x28
    net = deconv2DAct(net, prefix = "g4", actType = finalAct, iShape = Shape(ngf, 14, 14),
                        oShape = Shape(oShape.toArray.takeRight(3)), kShape = (4, 4))
    net
  }

  // DCGAN that generates 32x32 images.
  def dcgan32x32(oShape: Shape, finalAct: String, ngf: Int = 128): Symbol = {
    assert(oShape(oShape.length - 1) == 32)
    assert(oShape(oShape.length - 2) == 32)
    val code = Symbol.Variable("code")
    var net = Symbol.FullyConnected("g1")(Map("data" -> code, "num_hidden" -> 4 * 4 * ngf * 4, "no_bias" -> true))
    net = Symbol.Activation("gact1")(Map("data" -> net, "act_type" -> "relu"))
    // 4 x 4
    net = Symbol.Reshape()(Map("data" -> net, "shape" -> s"(-1, ${ngf * 4}, 4, 4)"))
    // 8 x 8
    net = deconv2DBnRelu(net, prefix = "g2", iShape = Shape(ngf * 4, 4, 4) , oShape = Shape(ngf * 2, 8, 8), kShape = (4, 4))
    // 16x16
    net = deconv2DBnRelu(net, iShape = Shape(ngf * 2, 8, 8), oShape = Shape(ngf, 16, 16), kShape = (4, 4), prefix = "g3")
    // 32 x 32
    net = deconv2DAct(net, prefix = "g4", actType = finalAct, iShape = Shape(ngf, 16, 16),
        oShape = Shape(oShape.toArray.takeRight(3)), kShape = (4, 4))
    net
  }

}
