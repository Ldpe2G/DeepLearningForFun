package models

import ml.dmlc.mxnet.Symbol
import ml.dmlc.mxnet.Shape

/**
 * @author Depeng Liang
 */
object Architectures {

    def defineUnet(outputNC: Int, ngf: Int): Symbol = {
    val data = Symbol.Variable("gData")
     var e1 = Symbol.Convolution()()(Map("data" -> data, "num_filter" -> ngf,
        "kernel" -> "(4, 4)", "stride" -> "(2, 2)", "pad" -> "(1, 1)", "no_bias" -> true, "cudnn_off" -> true))
     var e2 = Symbol.LeakyReLU()()(Map("data" -> e1, "slope" -> 0.2f))
     e2 = Symbol.Convolution()()(Map("data" -> e2, "num_filter" -> ngf * 2,
        "kernel" -> "(4, 4)", "stride" -> "(2, 2)", "pad" -> "(1, 1)", "no_bias" -> true, "cudnn_off" -> true))
     e2 = Symbol.BatchNorm()()(Map("data" -> e2))
     var e3 = Symbol.LeakyReLU()()(Map("data" -> e2, "slope" -> 0.2f))
     e3 = Symbol.Convolution()()(Map("data" -> e3, "num_filter" -> ngf * 4,
        "kernel" -> "(4, 4)", "stride" -> "(2, 2)", "pad" -> "(1, 1)", "no_bias" -> true, "cudnn_off" -> true))
     e3 = Symbol.BatchNorm()()(Map("data" -> e3))
     var e4 = Symbol.LeakyReLU()()(Map("data" -> e3, "slope" -> 0.2f))
     e4 = Symbol.Convolution()()(Map("data" -> e4, "num_filter" -> ngf * 8,
        "kernel" -> "(4, 4)", "stride" -> "(2, 2)", "pad" -> "(1, 1)", "no_bias" -> true, "cudnn_off" -> true))
     e4 = Symbol.BatchNorm()()(Map("data" -> e4))
     var e5 = Symbol.LeakyReLU()()(Map("data" -> e4, "slope" -> 0.2f))
     e5 = Symbol.Convolution()()(Map("data" -> e5, "num_filter" -> ngf * 8,
        "kernel" -> "(4, 4)", "stride" -> "(2, 2)", "pad" -> "(1, 1)", "no_bias" -> true, "cudnn_off" -> true))
     e5 = Symbol.BatchNorm()()(Map("data" -> e5))
     var e6 = Symbol.LeakyReLU()()(Map("data" -> e5, "slope" -> 0.2f))
     e6 = Symbol.Convolution()()(Map("data" -> e6, "num_filter" -> ngf * 8,
        "kernel" -> "(4, 4)", "stride" -> "(2, 2)", "pad" -> "(1, 1)", "no_bias" -> true, "cudnn_off" -> true))
     e6 = Symbol.BatchNorm()()(Map("data" -> e6))
     var e7 = Symbol.LeakyReLU()()(Map("data" -> e6, "slope" -> 0.2f))
     e7 = Symbol.Convolution()()(Map("data" -> e7, "num_filter" -> ngf * 8,
        "kernel" -> "(4, 4)", "stride" -> "(2, 2)", "pad" -> "(1, 1)", "no_bias" -> true, "cudnn_off" -> true))
    e7 = Symbol.BatchNorm()()(Map("data" -> e7))
    var e8 = Symbol.LeakyReLU()()(Map("data" -> e7, "slope" -> 0.2f))
    e8 = Symbol.Convolution()()(Map("data" -> e8, "num_filter" -> ngf * 8,
        "kernel" -> "(4, 4)", "stride" -> "(2, 2)", "pad" -> "(1, 1)", "no_bias" -> true, "cudnn_off" -> true))
    e8 = Symbol.BatchNorm()()(Map("data" -> e8))

    var d1_ = Symbol.Activation()()(Map("data" -> e8, "act_type" -> "relu"))
    d1_ = Symbol.Deconvolution()()(Map("data" -> d1_, "num_filter" -> ngf * 8,
            "kernel" -> "(4, 4)", "stride" -> "(2, 2)", "pad" -> "(1, 1)", "cudnn_off" -> true))    
    d1_ = Symbol.BatchNorm()()(Map("data" -> d1_))
    d1_ = Symbol.Dropout()()(Map("data" -> d1_, "p" -> 0.5f))
    
    d1_ = Symbol.Concat()(d1_, e7)(Map("dim" -> 1))
    var d2_ = Symbol.Activation()()(Map("data" -> d1_, "act_type" -> "relu"))
    d2_ = Symbol.Deconvolution()()(Map("data" -> d2_, "num_filter" -> ngf * 8,
            "kernel" -> "(4, 4)", "stride" -> "(2, 2)", "pad" -> "(1, 1)", "cudnn_off" -> true))    
    d2_ = Symbol.BatchNorm()()(Map("data" -> d2_))
    d2_ = Symbol.Dropout()()(Map("data" -> d2_, "p" -> 0.5f))

    d2_ = Symbol.Concat()(d2_, e6)(Map("dim" -> 1))
    var d3_ = Symbol.Activation()()(Map("data" -> d2_, "act_type" -> "relu"))
    d3_ = Symbol.Deconvolution()()(Map("data" -> d3_, "num_filter" -> ngf * 8,
            "kernel" -> "(4, 4)", "stride" -> "(2, 2)", "pad" -> "(1, 1)", "cudnn_off" -> true))    
    d3_ = Symbol.BatchNorm()()(Map("data" -> d3_))
    d3_ = Symbol.Dropout()()(Map("data" -> d3_, "p" -> 0.5f))

    d3_ = Symbol.Concat()(d3_, e5)(Map("dim" -> 1))
    var d4_ = Symbol.Activation()()(Map("data" -> d3_, "act_type" -> "relu"))
    d4_ = Symbol.Deconvolution()()(Map("data" -> d4_, "num_filter" -> ngf  * 8,
            "kernel" -> "(4, 4)", "stride" -> "(2, 2)", "pad" -> "(1, 1)", "cudnn_off" -> true))    
    d4_ = Symbol.BatchNorm()()(Map("data" -> d4_))

    d4_ = Symbol.Concat()(d4_, e4)(Map("dim" -> 1))
    var d5_ = Symbol.Activation()()(Map("data" -> d4_, "act_type" -> "relu"))
    d5_ = Symbol.Deconvolution()()(Map("data" -> d5_, "num_filter" -> ngf  * 4,
            "kernel" -> "(4, 4)", "stride" -> "(2, 2)", "pad" -> "(1, 1)", "cudnn_off" -> true))    
    d5_ = Symbol.BatchNorm()()(Map("data" -> d5_))

    d5_ = Symbol.Concat()(d5_, e3)(Map("dim" -> 1))
    var d6_ = Symbol.Activation()()(Map("data" -> d5_, "act_type" -> "relu"))
    d6_ = Symbol.Deconvolution()()(Map("data" -> d6_, "num_filter" -> ngf * 2,
            "kernel" -> "(4, 4)", "stride" -> "(2, 2)", "pad" -> "(1, 1)", "cudnn_off" -> true))    
    d6_ = Symbol.BatchNorm()()(Map("data" -> d6_))

    d6_ = Symbol.Concat()(d6_, e2)(Map("dim" -> 1))
    var d7_ = Symbol.Activation()()(Map("data" -> d6_, "act_type" -> "relu"))
    d7_ = Symbol.Deconvolution()()(Map("data" -> d7_, "num_filter" -> ngf,
            "kernel" -> "(4, 4)", "stride" -> "(2, 2)", "pad" -> "(1, 1)", "cudnn_off" -> true))    
    d7_ = Symbol.BatchNorm()()(Map("data" -> d7_))

    d7_ = Symbol.Concat()(d7_, e1)(Map("dim" -> 1))
    var d8_ = Symbol.Activation()()(Map("data" -> d7_, "act_type" -> "relu"))
    d8_ = Symbol.Deconvolution()()(Map("data" -> d8_, "num_filter" -> outputNC,
            "kernel" -> "(4, 4)", "stride" -> "(2, 2)", "pad" -> "(1, 1)", "cudnn_off" -> true))

    var netG = Symbol.Activation()()(Map("data" -> d8_, "act_type" -> "tanh"))
    netG
  }
    
  def defineDNLayers(ndf: Int, nLayers: Int): Symbol = {
    val data = Symbol.Variable("dData")

    var netD = data
    for (i <- 0 until nLayers) {
      if (i == 0) {
        netD = Symbol.Convolution()()(Map("data" -> netD, "num_filter" -> ndf,
          "kernel" -> "(4, 4)", "stride" -> "(2, 2)", "pad" -> "(1, 1)", "no_bias" -> true, "cudnn_off" -> true))
        netD = Symbol.LeakyReLU()()(Map("data" -> netD, "slope" -> 0.2f))
      } else {
        val nfMul = Math.min(Math.pow(2, i).toInt, 8)
        netD = Symbol.Convolution()()(Map("data" -> netD, "num_filter" -> ndf * nfMul,
          "kernel" -> "(4, 4)", "stride" -> "(2, 2)", "pad" -> "(1, 1)", "no_bias" -> true, "cudnn_off" -> true))
        netD = Symbol.BatchNorm()()(Map("data" -> netD))
        netD = Symbol.LeakyReLU()()(Map("data" -> netD, "slope" -> 0.2f))
      }
    }

    val nfMul = Math.min(Math.pow(2, nLayers).toInt, 8)
    netD = Symbol.Convolution()()(Map("data" -> netD, "num_filter" -> ndf * nfMul,
      "kernel" -> s"(4, 4)", "stride" -> "(1, 1)", "pad" -> "(1, 1)", "no_bias" -> true, "cudnn_off" -> true))
    netD = Symbol.BatchNorm()()(Map("data" -> netD))
    netD = Symbol.LeakyReLU()()(Map("data" -> netD, "slope" -> 0.2f))

    netD = Symbol.Convolution()()(Map("data" -> netD, "num_filter" -> 1,
      "kernel" -> "(4, 4)", "stride" -> "(1, 1)", "pad" -> "(1, 1)", "no_bias" -> true, "cudnn_off" -> true))

    netD = Symbol.LogisticRegressionOutput("dloss")()(Map("data" -> netD))
    netD
  }

  def getAbsLoss(): Symbol = {
    val origin = Symbol.Variable("origin")
    val rec = Symbol.Variable("rec")
    val diff = origin - rec
    val abs = Symbol.abs()()(Map("data" -> diff))
    val mean = Symbol.mean()()(Map("data" -> abs))
    Symbol.MakeLoss()()(Map("data" -> mean))
  }
}