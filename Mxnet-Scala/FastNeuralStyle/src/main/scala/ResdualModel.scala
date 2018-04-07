
import org.apache.mxnet.Shape
import org.apache.mxnet.Context
import org.apache.mxnet.Xavier
import org.apache.mxnet.Symbol

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
    val conv = Symbol.Convolution()()(Map("data" -> data,
      "num_filter" -> numFilter, "kernel" -> s"$kernel",
      "stride" -> s"$stride", "pad" -> s"$pad"))
    val bn = Symbol.BatchNorm()()(Map("data" -> conv))
    val relu = convType match {
      case ConvWitAct => Symbol.Activation()()(Map("data" -> bn, "act_type" -> actType))
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

    var conv = Symbol.Convolution()()(Map(
      "data" -> data, "num_filter" -> 32, "kernel" -> "(9, 9)",
      "stride" -> "(1, 1)", "pad" -> "(4, 4)", "no_bias" -> false))
    var bn = Symbol.BatchNorm()()(Map("data" -> conv))
    var relu = Symbol.Activation()()(Map("data" -> bn, "act_type" -> "relu"))
    
    conv = Symbol.Convolution()()(Map(
      "data" -> conv, "num_filter" -> 64, "kernel" -> "(3, 3)",
      "stride" -> "(2, 2)", "pad" -> "(1, 1)", "no_bias" -> false))
    bn = Symbol.BatchNorm()()(Map("data" -> conv))
    relu = Symbol.Activation()()(Map("data" -> bn, "act_type" -> "relu"))
    
    conv = Symbol.Convolution()()(Map(
      "data" -> conv, "num_filter" -> 128, "kernel" -> "(3, 3)",
      "stride" -> "(2, 2)", "pad" -> "(1, 1)", "no_bias" -> false))
    bn = Symbol.BatchNorm()()(Map("data" -> conv))
    relu = Symbol.Activation()()(Map("data" -> bn, "act_type" -> "relu"))
      
    var res = buildResBlock(conv, 128)
    res = buildResBlock(res, 128)
    res = buildResBlock(res, 128)
    res = buildResBlock(res, 128)
    res = buildResBlock(res, 128)
    
    var deConv = Symbol.Deconvolution()()(Map("data" -> res,
        "target_shape" -> s"(${imHw._1 / 2}, ${imHw._2 / 2})", "num_filter" -> 64,
        "kernel" -> "(4, 4)", "stride" -> "(2, 2)", "no_bias" -> true))
    bn = Symbol.BatchNorm()()(Map("data" -> deConv))
    relu = Symbol.Activation()()(Map("data" -> bn, "act_type" -> "relu"))
    
    deConv = Symbol.Deconvolution()()(Map("data" -> relu,
        "target_shape" -> s"(${imHw._1}, ${imHw._2})", "num_filter" -> 32,
        "kernel" -> "(3, 3)", "stride" -> "(2, 2)", "no_bias" -> true))
    bn = Symbol.BatchNorm()()(Map("data" -> deConv))
    relu = Symbol.Activation()()(Map("data" -> bn, "act_type" -> "relu"))

    conv = Symbol.Convolution()()(Map(
      "data" -> relu, "num_filter" -> 3, "kernel" -> "(9, 9)",
      "stride" -> "(1, 1)", "pad" -> "(4, 4)", "no_bias" -> false))
      var out = Symbol.Activation()()(Map("data" -> conv, "act_type" -> "tanh"))
      
     val rawOut = (out * 128) + 128
     val norm = Symbol.SliceChannel()(rawOut)(Map("num_outputs" -> 3))
     val rCh = norm.get(0) - 123.68f
     val gCh = norm.get(1) - 116.779f
     val bCh = norm.get(2) - 103.939f
     val normOut = Symbol.Concat()(rCh, gCh, bCh)()
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