package models

import ml.dmlc.mxnet.Symbol
import ml.dmlc.mxnet.Shape

/**
 * @author Depeng Liang
 */
object Architectures {

  sealed trait ConvType
  case object ConvWithoutAct extends ConvType
  case object ConvWitAct extends ConvType

  def convFactory(data: Symbol, numFilter: Int, kernel: (Int, Int), stride: (Int, Int),
      norm: Map[String, Any] => Symbol, actType: String = "relu", convType: ConvType = ConvWitAct): Symbol = {
    val conv = Symbol.Convolution()()(Map("data" -> data,
      "num_filter" -> numFilter, "kernel" -> s"$kernel", "stride" -> s"$stride", "pad" -> "(1, 1)", "no_bias" -> true))
    val bn = norm(Map("data" -> conv))
    val act = convType match {
      case ConvWitAct => Symbol.Activation()()(Map("data" -> bn, "act_type" -> actType))
      case ConvWithoutAct => bn
    }
    act
  }

  def buildResBlock(data: Symbol, numFilter: Int, norm: Map[String, Any] => Symbol): Symbol = {
    val conv1 = convFactory(data = data, numFilter = numFilter, kernel = (3, 3), stride = (1, 1),
        norm = norm, actType = "relu", convType = ConvWitAct)
    val conv2 = convFactory(data = conv1, numFilter = numFilter, kernel = (3, 3), stride = (1, 1),
        norm = norm, convType = ConvWithoutAct)
    data + conv2
  }
  
  def getNormFun(prefix: String, normType: String = "batch"): Map[String, Any] => Symbol = {
    if (normType == "batch") Symbol.BatchNorm()() _
    else if (normType == "instance") Symbol.InstanceNorm()() _
    else null
  }

  def defineGResNet6Blocks(outputNC: Int = 3, ngf: Int = 32, normType: String = "batch"): Symbol = {
    val ks = 3
    val f = 7
    val p = (f - 1) / 2

    val data = Symbol.Variable("gData")

    val norm = getNormFun(normType)

    var e1 = Symbol.Convolution()()(Map("data" -> data, "num_filter" -> ngf,
        "kernel" -> s"($f, $f)", "stride" -> "(1, 1)", "pad" -> s"($p, $p)", "no_bias" -> true))
    e1 = norm(Map("data" -> e1))
    e1 = Symbol.Activation()()(Map("data" -> e1, "act_type" -> "relu"))

    var e2 = Symbol.Convolution()()(Map("data" -> e1, "num_filter" -> ngf * 2,
        "kernel" -> s"($ks, $ks)", "stride" -> "(2, 2)", "pad" -> "(1, 1)", "no_bias" -> true))
    e2 = norm(Map("data" -> e2))
    e2 = Symbol.Activation()()(Map("data" -> e2, "act_type" -> "relu"))

    var e3 = Symbol.Convolution()()(Map("data" -> e2, "num_filter" -> ngf * 4,
        "kernel" -> s"($ks, $ks)", "stride" -> "(2, 2)", "pad" -> "(1, 1)", "no_bias" -> true))
    e3 = norm(Map("data" -> e3))
    e3 = Symbol.Activation()()(Map("data" -> e3, "act_type" -> "relu"))

    var d1 = buildResBlock(e3, ngf * 4, norm)
    d1 = buildResBlock(d1, ngf * 4, norm)
    d1 = buildResBlock(d1, ngf * 4, norm)
    d1 = buildResBlock(d1, ngf * 4, norm)
    d1 = buildResBlock(d1, ngf * 4, norm)
    d1 = buildResBlock(d1, ngf * 4, norm)

    var d2 = Symbol.Deconvolution()()(Map("data" -> d1, "kernel" -> s"($ks, $ks)", "num_filter" -> ngf * 2,
      "stride" -> "(2, 2)", "pad" -> "(1, 1)", "adj" -> "(1, 1)", "no_bias" -> true
    ))
    d2 = norm(Map("data" -> d2))
    d2 = Symbol.Activation()()(Map("data" -> d2, "act_type" -> "relu"))

    var d3 = Symbol.Deconvolution()()(Map("data" -> d2, "kernel" -> s"($ks, $ks)", "num_filter" -> ngf,
      "stride" -> "(2, 2)", "pad" -> "(1, 1)", "adj" -> "(1, 1)", "no_bias" -> true
    ))
    d3 = norm(Map("data" -> d3))
    d3 = Symbol.Activation()()(Map("data" -> d3, "act_type" -> "relu"))

    var d4 = Symbol.Convolution()()(Map("data" -> d3, "num_filter" -> outputNC,
        "kernel" -> s"($f, $f)", "stride" -> "(1, 1)", "pad" -> s"($p, $p)", "no_bias" -> true))
    val netG = Symbol.Activation()()(Map("data" -> d4, "act_type" -> "tanh"))

    netG
  }

  def defineDNLayers(ndf: Int, nLayers: Int = 3, kw: Int = 4,
      dropoutRatio: Float = 0f, normType: String = "batch"): Symbol = {
    val padW = Math.ceil((kw - 1) / 2).toInt

    val norm = getNormFun(normType)

    val data = Symbol.Variable("dData")

    var netD = data
    for (i <- 0 until nLayers) {
      if (i == 0) {
        netD = Symbol.Convolution()()(Map("data" -> netD, "num_filter" -> ndf,
          "kernel" -> s"($kw, $kw)", "stride" -> "(2, 2)", "pad" -> s"($padW, $padW)", "no_bias" -> true))
        netD = Symbol.LeakyReLU()()(Map("data" -> netD, "slope" -> 0.2f))
      } else {
        val nfMul = Math.min(Math.pow(2, i).toInt, 8)
        netD = Symbol.Convolution()()(Map("data" -> netD, "num_filter" -> ndf * nfMul,
          "kernel" -> s"($kw, $kw)", "stride" -> "(2, 2)", "pad" -> s"($padW, $padW)", "no_bias" -> true))
        netD = norm(Map("data" -> netD))
        if (dropoutRatio > 0f) Symbol.Dropout()()(Map("data" -> netD, "p" -> dropoutRatio))
        netD = Symbol.LeakyReLU()()(Map("data" -> netD, "slope" -> 0.2f))
      }
    }

    val nfMul = Math.min(Math.pow(2, nLayers).toInt, 8)
    netD = Symbol.Convolution()()(Map("data" -> netD, "num_filter" -> ndf * nfMul,
      "kernel" -> s"($kw, $kw)", "stride" -> "(1, 1)", "pad" -> s"($padW, $padW)", "no_bias" -> true))
    netD = norm(Map("data" -> netD))
    netD = Symbol.LeakyReLU()()(Map("data" -> netD, "slope" -> 0.2f))

    netD = Symbol.Convolution()()(Map("data" -> netD, "num_filter" -> 1,
      "kernel" -> s"($kw, $kw)", "stride" -> "(1, 1)", "pad" -> s"($padW, $padW)", "no_bias" -> true))

    netD
  }

  def getMSELoss(div: Float): Symbol = {
    val data = Symbol.Variable("datas")
    val label = Symbol.Variable("labels")
    val diff = data - label
    val square = Symbol.square()()(Map("data" -> diff))
    val mean = Symbol.sum()()(Map("data" -> square))
    Symbol.MakeLoss()()(Map("data" -> mean))
  }

  def getAbsLoss(div: Float): Symbol = {
    val origin = Symbol.Variable("origin")
    val rec = Symbol.Variable("rec")
    val diff = origin - rec
    val abs = Symbol.abs()()(Map("data" -> diff))
    val mean = Symbol.sum()()(Map("data" -> abs))
    Symbol.MakeLoss()()(Map("data" -> mean))
  }
}