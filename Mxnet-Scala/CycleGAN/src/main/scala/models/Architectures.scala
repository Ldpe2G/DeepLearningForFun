package models

import org.apache.mxnet.Symbol
import org.apache.mxnet.Shape
import org.apache.mxnet.util.OptionConversion._

/**
 * @author Depeng Liang
 */
object Architectures {

  sealed trait ConvType
  case object ConvWithoutAct extends ConvType
  case object ConvWitAct extends ConvType

  def convFactory(data: Symbol, numFilter: Int, kernel: (Int, Int), stride: (Int, Int), actType: String = "relu", convType: ConvType = ConvWitAct, normType: String = "batch"): Symbol = {
    val conv = Symbol.api.Convolution(data, num_filter = numFilter, kernel = Shape(kernel._1, kernel._2),
                                                                  stride = Shape(stride._1, stride._2), pad = Shape(1, 1), no_bias = true)
    val bn = if (normType == "batch") Symbol.api.BatchNorm(conv) else Symbol.api.InstanceNorm(conv)
    val act = convType match {
      case ConvWitAct => Symbol.api.Activation(bn, act_type = actType)
      case ConvWithoutAct => bn
    }
    act
  }

  def buildResBlock(data: Symbol, numFilter: Int, normType: String = "batch"): Symbol = {
    val conv1 = convFactory(data = data, numFilter = numFilter, kernel = (3, 3), stride = (1, 1), actType = "relu", convType = ConvWitAct, normType = normType)
    val conv2 = convFactory(data = conv1, numFilter = numFilter, kernel = (3, 3), stride = (1, 1), convType = ConvWithoutAct, normType = normType)
    data + conv2
  }

  def defineGResNet6Blocks(outputNC: Int = 3, ngf: Int = 32, normType: String = "batch"): Symbol = {
    val ks = 3
    val f = 7
    val p = (f - 1) / 2

    val data = Symbol.Variable("gData")

    var e1 = Symbol.api.Convolution(data, num_filter = ngf, kernel = Shape(f, f), stride = Shape(1, 1), pad = Shape(p, p), no_bias = true)
    e1 = if (normType == "batch") Symbol.api.BatchNorm(e1) else Symbol.api.InstanceNorm(e1)
    e1 = Symbol.api.relu(e1)

    var e2 = Symbol.api.Convolution(e1, num_filter = ngf * 2, kernel = Shape(ks, ks), stride = Shape(2, 2), pad = Shape(1, 1), no_bias = true)
    e2 = if (normType == "batch") Symbol.api.BatchNorm(e2) else Symbol.api.InstanceNorm(e2)
    e2 = Symbol.api.relu(e2)

    var e3 = Symbol.api.Convolution(e2, num_filter = ngf * 4, kernel = Shape(ks, ks), stride = Shape(2, 2), pad = Shape(1, 1), no_bias = true)
    e3 = if (normType == "batch") Symbol.api.BatchNorm(e3) else Symbol.api.InstanceNorm(e3)
    e3 = Symbol.api.relu(e3)

    var d1 = buildResBlock(e3, ngf * 4, normType)
    d1 = buildResBlock(d1, ngf * 4, normType)
    d1 = buildResBlock(d1, ngf * 4, normType)
    d1 = buildResBlock(d1, ngf * 4, normType)
    d1 = buildResBlock(d1, ngf * 4, normType)
    d1 = buildResBlock(d1, ngf * 4, normType)

    var d2 = Symbol.api.Deconvolution(d1, kernel = Shape(ks, ks), num_filter = ngf * 2, stride = Shape(2, 2), pad = Shape(1, 1), adj = Shape(1, 1), no_bias = true)
    d2 = if (normType == "batch") Symbol.api.BatchNorm(d2) else Symbol.api.InstanceNorm(d2)
    d2 = Symbol.api.relu(d2)

    var d3 = Symbol.api.Deconvolution(d2, kernel = Shape(ks, ks), num_filter = ngf, stride = Shape(2, 2), pad = Shape(1, 1), adj = Shape(1, 1), no_bias =  true)
    d3 = if (normType == "batch") Symbol.api.BatchNorm(d3) else Symbol.api.InstanceNorm(d3)
    d3 = Symbol.Activation()()(Map("data" -> d3, "act_type" -> "relu"))

    var d4 = Symbol.api.Convolution(d3, num_filter = outputNC, kernel = Shape(f, f), stride = Shape(1, 1), pad = Shape(p, p), no_bias = true)
    val netG = Symbol.api.tanh(d4)

    netG
  }

  def defineDNLayers(ndf: Int, nLayers: Int = 3, kw: Int = 4,
      dropoutRatio: Float = 0f, normType: String = "batch"): Symbol = {
    val padW = Math.ceil((kw - 1) / 2).toInt

    val data = Symbol.Variable("dData")

    var netD = data
    for (i <- 0 until nLayers) {
      if (i == 0) {
        netD = Symbol.api.Convolution(netD, num_filter = ndf, kernel = Shape(kw, kw), stride = Shape(2, 2), pad = Shape(padW, padW), no_bias = true)
        netD = Symbol.api.LeakyReLU(netD, slope = 0.2f)
      } else {
        val nfMul = Math.min(Math.pow(2, i).toInt, 8)
        netD = Symbol.api.Convolution(netD, num_filter = ndf * nfMul, kernel = Shape(kw, kw), stride = Shape(2, 2), pad = Shape(padW, padW), no_bias = true)
        netD =  if (normType == "batch") Symbol.api.BatchNorm(netD) else Symbol.api.InstanceNorm(netD)
        if (dropoutRatio > 0f) Symbol.api.Dropout(netD, p = dropoutRatio)
        netD = Symbol.api.LeakyReLU(netD, slope = 0.2f)
      }
    }

    val nfMul = Math.min(Math.pow(2, nLayers).toInt, 8)
    netD = Symbol.api.Convolution(netD, num_filter = ndf * nfMul, kernel = Shape(kw, kw), stride = Shape(1, 1), pad = Shape(padW, padW), no_bias = true)
    netD = if (normType == "batch") Symbol.api.BatchNorm(netD) else Symbol.api.InstanceNorm(netD)
    netD = Symbol.api.LeakyReLU(netD, slope = 0.2f)

    netD = Symbol.api.Convolution(netD, num_filter = 1, kernel = Shape(kw, kw), stride = Shape(1, 1), pad = Shape(padW, padW), no_bias = true)

    netD
  }

  def getMSELoss(div: Float): Symbol = {
    val data = Symbol.Variable("datas")
    val label = Symbol.Variable("labels")
    val diff = data - label
    val square = Symbol.api.square(diff)
    val mean = Symbol.api.sum(square)
    Symbol.api.MakeLoss(mean)
  }

  def getAbsLoss(div: Float): Symbol = {
    val origin = Symbol.Variable("origin")
    val rec = Symbol.Variable("rec")
    val diff = origin - rec
    val abs = Symbol.api.abs(diff)
    val mean = Symbol.api.sum(abs)
    Symbol.api.MakeLoss(mean)
  }
}