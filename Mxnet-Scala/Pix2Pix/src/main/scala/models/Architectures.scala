package models

import org.apache.mxnet.Symbol
import org.apache.mxnet.Shape
import org.apache.mxnet.util.OptionConversion._

/**
 * @author Depeng Liang
 */
object Architectures {

    def defineUnet(outputNC: Int, ngf: Int): Symbol = {
    val data = Symbol.Variable("gData")
     var e1 = Symbol.api.Convolution(data, num_filter = ngf,
        kernel = Shape(4, 4), stride = Shape(2, 2), pad = Shape(1, 1), no_bias = true, cudnn_off = true)
     var e2 = Symbol.api.LeakyReLU(e1, slope = 0.2f)
     e2 = Symbol.api.Convolution(e2, num_filter = ngf * 2,
        kernel = Shape(4, 4), stride = Shape(2, 2), pad = Shape(1, 1), no_bias = true, cudnn_off = true)
     e2 = Symbol.api.BatchNorm(e2)
     var e3 = Symbol.api.LeakyReLU(e2, slope = 0.2f)
     e3 = Symbol.api.Convolution(e3, num_filter = ngf * 4,
        kernel = Shape(4, 4), stride = Shape(2, 2), pad = Shape(1, 1), no_bias = true, cudnn_off = true)
     e3 = Symbol.api.BatchNorm(e3)
     var e4 = Symbol.api.LeakyReLU(e3, slope = 0.2f)
     e4 = Symbol.api.Convolution(e4, num_filter = ngf * 8,
        kernel = Shape(4, 4), stride = Shape(2, 2), pad = Shape(1, 1), no_bias = true, cudnn_off = true)
     e4 = Symbol.api.BatchNorm(e4)
     var e5 = Symbol.LeakyReLU()()(Map("data" -> e4, "slope" -> 0.2f))
     e5 = Symbol.api.Convolution(e5, num_filter = ngf * 8,
        kernel = Shape(4, 4), stride = Shape(2, 2), pad = Shape(1, 1), no_bias = true, cudnn_off = true)
     e5 = Symbol.api.BatchNorm(e5)
     var e6 = Symbol.api.LeakyReLU(e5, slope = 0.2f)
     e6 = Symbol.api.Convolution(e6, num_filter = ngf * 8,
        kernel = Shape(4, 4), stride = Shape(2, 2), pad = Shape(1, 1), no_bias = true, cudnn_off = true)
     e6 = Symbol.api.BatchNorm(e6)
     var e7 = Symbol.api.LeakyReLU(e6, slope = 0.2f)
     e7 = Symbol.api.Convolution(e7, num_filter = ngf * 8,
        kernel = Shape(4, 4), stride = Shape(2, 2), pad = Shape(1, 1), no_bias = true, cudnn_off = true)
    e7 = Symbol.api.BatchNorm(e7)
    var e8 = Symbol.api.LeakyReLU(e7, slope = 0.2f)
    e8 = Symbol.api.Convolution(e8, num_filter = ngf * 8,
        kernel = Shape(4, 4), stride = Shape(2, 2), pad = Shape(1, 1), no_bias = true, cudnn_off = true)
    e8 = Symbol.api.BatchNorm(e8)

    var d1_ = Symbol.api.relu(e8)
    d1_ = Symbol.api.Deconvolution(d1_, num_filter = ngf * 8,
            kernel = Shape(4, 4), stride = Shape(2, 2), pad = Shape(1, 1), cudnn_off = true)    
    d1_ = Symbol.api.BatchNorm(d1_)
    d1_ = Symbol.api.Dropout(d1_, p = 0.5f)
    
    d1_ = Symbol.api.Concat(Array(d1_, e7), num_args = 2, dim = 1)
    var d2_ = Symbol.api.relu(d1_)
    d2_ = Symbol.api.Deconvolution(d2_, num_filter = ngf * 8,
            kernel = Shape(4, 4), stride = Shape(2, 2), pad = Shape(1, 1), cudnn_off = true)    
    d2_ = Symbol.api.BatchNorm(d2_)
    d2_ = Symbol.api.Dropout(d2_, p = 0.5f)

    d2_ = Symbol.api.Concat(Array(d2_, e6), num_args = 2, dim = 1)
    var d3_ = Symbol.api.relu(d2_)
    d3_ = Symbol.api.Deconvolution(d3_, num_filter = ngf * 8,
            kernel = Shape(4, 4), stride = Shape(2, 2), pad = Shape(1, 1), cudnn_off = true)    
    d3_ = Symbol.api.BatchNorm(d3_)
    d3_ = Symbol.api.Dropout(d3_, p = 0.5f)

    d3_ = Symbol.api.Concat(Array(d3_, e5), num_args = 2, dim = 1)
    var d4_ = Symbol.api.relu(d3_)
    d4_ = Symbol.api.Deconvolution(d4_, num_filter = ngf  * 8,
            kernel = Shape(4, 4), stride = Shape(2, 2), pad = Shape(1, 1), cudnn_off = true)    
    d4_ = Symbol.api.BatchNorm(d4_)

    d4_ = Symbol.api.Concat(Array(d4_, e4), num_args = 2, dim = 1)
    var d5_ = Symbol.api.relu(d4_)
    d5_ = Symbol.api.Deconvolution(d5_, num_filter = ngf  * 4,
            kernel = Shape(4, 4), stride = Shape(2, 2), pad = Shape(1, 1), cudnn_off = true)    
    d5_ = Symbol.api.BatchNorm(d5_)

    d5_ = Symbol.api.Concat(Array(d5_, e3), num_args = 2, dim = 1)
    var d6_ = Symbol.api.relu(d5_)
    d6_ = Symbol.api.Deconvolution(d6_, num_filter = ngf * 2,
            kernel = Shape(4, 4), stride = Shape(2, 2), pad = Shape(1, 1), cudnn_off = true)    
    d6_ = Symbol.api.BatchNorm(d6_)

    d6_ = Symbol.api.Concat(Array(d6_, e2), num_args = 2, dim = 1)
    var d7_ = Symbol.api.relu(d6_)
    d7_ = Symbol.api.Deconvolution(d7_, num_filter = ngf,
            kernel = Shape(4, 4), stride = Shape(2, 2), pad = Shape(1, 1), cudnn_off = true)    
    d7_ = Symbol.api.BatchNorm(d7_)

    d7_ = Symbol.api.Concat(Array(d7_, e1), num_args = 2, dim = 1)
    var d8_ = Symbol.api.relu(d7_)
    d8_ = Symbol.api.Deconvolution(d8_, num_filter = outputNC,
            kernel = Shape(4, 4), stride = Shape(2, 2), pad = Shape(1, 1), cudnn_off = true)

    var netG = Symbol.api.tanh(d8_)
    netG
  }
    
  def defineDNLayers(ndf: Int, nLayers: Int): Symbol = {
    val data = Symbol.Variable("dData")

    var netD = data
    for (i <- 0 until nLayers) {
      if (i == 0) {
        netD = Symbol.api.Convolution(netD, num_filter = ndf,
          kernel = Shape(4, 4), stride = Shape(2, 2), pad = Shape(1, 1), no_bias = true, cudnn_off = true)
        netD = Symbol.api.LeakyReLU(netD, slope = 0.2f)
      } else {
        val nfMul = Math.min(Math.pow(2, i).toInt, 8)
        netD = Symbol.api.Convolution(netD, num_filter = ndf * nfMul,
          kernel = Shape(4, 4), stride = Shape(2, 2), pad = Shape(1, 1), no_bias = true, cudnn_off = true)
        netD = Symbol.api.BatchNorm(netD)
        netD = Symbol.api.LeakyReLU(netD, slope = 0.2f)
      }
    }

    val nfMul = Math.min(Math.pow(2, nLayers).toInt, 8)
    netD = Symbol.api.Convolution(netD, num_filter = ndf * nfMul,
      kernel = Shape(4, 4), stride = Shape(1, 1), pad = Shape(1, 1), no_bias = true, cudnn_off = true)
    netD = Symbol.api.BatchNorm(netD)
    netD = Symbol.api.LeakyReLU(netD, slope = 0.2f)

    netD = Symbol.api.Convolution(netD, num_filter = 1,
      kernel = Shape(4, 4), stride = Shape(1, 1), pad = Shape(1, 1), no_bias = true, cudnn_off = true)

    netD = Symbol.api.LogisticRegressionOutput(netD, name = "dloss")
    netD
  }
  
  def getAbsLoss(): Symbol = {
    val origin = Symbol.Variable("origin")
    val rec = Symbol.Variable("rec")
    val diff = origin - rec
    val abs = Symbol.api.abs(diff)
    val mean = Symbol.api.mean(abs)
    Symbol.api.MakeLoss(mean)
  }
}