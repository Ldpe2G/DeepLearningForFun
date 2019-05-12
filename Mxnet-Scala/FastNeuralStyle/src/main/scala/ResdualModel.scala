
import org.apache.mxnet.Shape
import org.apache.mxnet.Context
import org.apache.mxnet.Xavier
import org.apache.mxnet.Symbol
import org.apache.mxnet.util.OptionConversion._

/**
 * @author Depeng Liang
 */
object ResdualModel {
  
  sealed trait ConvType
  case object ConvWithoutAct extends ConvType
  case object ConvWitAct extends ConvType

  def convFactory(data: Symbol, numFilter: Int, kernel: (Int, Int),
      stride: (Int, Int), pad: (Int, Int), actType: String = "relu",
      convType: ConvType = ConvWitAct): Symbol = {
    val conv = Symbol.api.Convolution(data, num_filter = numFilter, kernel = Shape(kernel._1, kernel._2), stride = Shape(stride._1, stride._2), pad = Shape(pad._1, pad._2))
    val bn = Symbol.api.BatchNorm(conv)
    val relu = convType match {
      case ConvWitAct => Symbol.api.Activation(bn, act_type = actType)
      case ConvWithoutAct => bn
    }
    relu
  }
  
  def buildResBlock(data: Symbol, numFilter: Int): Symbol = {
    val conv1 = convFactory(data = data, numFilter = numFilter, kernel = (3, 3),
        stride = (1, 1), pad = (1, 1), actType = "relu", convType = ConvWitAct)
    val conv2 = convFactory(data = conv1, numFilter = numFilter, kernel = (3, 3),
        stride = (1, 1), pad = (1, 1), convType = ConvWithoutAct)
    val newData = data + conv2
    newData
  }
  
  def getStyleTransferNetwork(prefix: String, imHw: (Int, Int)): Symbol = {
    val data = Symbol.Variable(s"${prefix}_data")

    var conv = Symbol.api.Convolution(data, num_filter = 32, kernel = Shape(9, 9), stride = Shape(1, 1), pad = Shape(4, 4), no_bias = false)
    var bn = Symbol.api.BatchNorm(conv)
    var relu = Symbol.api.relu(bn)
    
    conv = Symbol.api.Convolution(conv, num_filter = 64, kernel = Shape(3, 3), stride = Shape(2, 2), pad = Shape(1, 1), no_bias = false)
    bn = Symbol.api.BatchNorm(conv)
    relu = Symbol.api.relu(bn)
    
    conv = Symbol.api.Convolution(conv, num_filter = 128, kernel = Shape(3, 3), stride = Shape(2, 2), pad = Shape(1, 1), no_bias = false)
    bn = Symbol.api.BatchNorm(conv)
    relu = Symbol.api.relu(bn)

    var res = buildResBlock(conv, 128)
    res = buildResBlock(res, 128)
    res = buildResBlock(res, 128)
    res = buildResBlock(res, 128)
    res = buildResBlock(res, 128)
    
    var deConv = Symbol.api.Deconvolution(res, target_shape = Shape(imHw._1 / 2, imHw._2 / 2), num_filter =  64, kernel = Shape(4, 4), stride = Shape(2, 2), no_bias = true)
    bn = Symbol.api.BatchNorm(deConv)
    relu = Symbol.api.relu(bn)
    
    deConv = Symbol.api.Deconvolution(relu, target_shape = Shape(imHw._1, imHw._2), num_filter = 32, kernel = Shape(3, 3), stride = Shape(2, 2), no_bias = true)
    bn = Symbol.api.BatchNorm(deConv)
    relu = Symbol.api.relu(bn)

    conv = Symbol.api.Convolution(relu, num_filter = 3, kernel = Shape(9, 9), stride = Shape(1, 1), pad = Shape(4, 4), no_bias = false)
    var out = Symbol.api.tanh(conv)

     val rawOut = (out * 128) + 128
     val norm = Symbol.api.SliceChannel(rawOut, num_outputs = 3)
     val rCh = norm.get(0) - 123.68f
     val gCh = norm.get(1) - 116.779f
     val bCh = norm.get(2) - 103.939f
     val normOut = Symbol.api.Concat(Array(rCh, gCh, bCh), num_args = 3)
      normOut
  }

  def getModule(prefix: String, dShape: Shape, ctx: Context,
      training: Boolean = true): Module = {
    val sym = getStyleTransferNetwork(prefix, (dShape(2), dShape(3)))
    val (dataShapes, forTraining) = {
      val dataShape = Map(s"${prefix}_data" -> dShape)
      (dataShape, training)
    }
    val mod = new Module(symbol = sym, context = ctx,
                         dataShapes = dataShapes,
                         initializer = new Xavier(magnitude = 2f),
                         forTraining = forTraining)
    mod
  }

}