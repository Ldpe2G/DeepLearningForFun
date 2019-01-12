import org.apache.mxnet._
import org.apache.mxnet.util.OptionConversion._
import QuantizeOps._

object InferQuanVGGSym {
  private def getQuanFeature(input: Symbol, layers: Array[Int], filters: Array[Int], quantizeLevel: Int): Symbol = {
    var internelLayer = input
    
    for (i <- 0 until layers.length) {
      for (j <- 0 until layers(i)) {
        val weight = Symbol.Variable(s"conv${i + 1}_${j + 1}_weight")
        val bias = Symbol.Variable(s"conv${i + 1}_${j + 1}_bias")
        val gamma = Symbol.Variable(s"bn${i + 1}_${j + 1}_gamma")
        val beta = Symbol.Variable(s"bn${i + 1}_${j + 1}_beta")
        val moving_mean = Symbol.Variable(s"bn${i + 1}_${j + 1}_moving_mean")
        val moving_var = Symbol.Variable(s"bn${i + 1}_${j + 1}_moving_var")
        
        Operator.register(s"merge_bn${i + 1}_${j + 1}", new MergeBNProp())
        val bnWargs = scala.collection.mutable.Map[String, Any](
            "weight" -> weight, "bias" -> bias,
            "gamma" -> Symbol.api.BlockGrad(gamma),
            "beta" -> Symbol.api.BlockGrad(beta),
            "moving_mean" -> Symbol.api.BlockGrad(moving_mean),
            "moving_var" -> Symbol.api.BlockGrad(moving_var)
        )
        val mergeBn: Symbol = Symbol.api.Custom(s"merge_bn${i + 1}_${j + 1}", 
            name = s"merge_bn${i + 1}_${j + 1}", kwargs = bnWargs)

        val scale = Symbol.Variable(s"conv${i + 1}_${j + 1}_in_act_scale_beta")
        var quanData = Symbol.api.round(Symbol.api.broadcast_mul(internelLayer, scale))
        quanData = Symbol.api.clip(data=quanData, -quantizeLevel, quantizeLevel - 1)
        
        val weightScale = Symbol.Variable(s"conv${i + 1}_${j + 1}_weight_scale_beta")
        var quanWeight = Symbol.api.round(Symbol.api.broadcast_mul(mergeBn.get(0), weightScale))
        quanWeight = Symbol.api.clip(data=quanWeight, -(quantizeLevel - 1), quantizeLevel - 1)
        
        var quanBias = Symbol.api.round(data=Symbol.api.broadcast_mul(mergeBn.get(1), weightScale * scale))
        var conv = Symbol.api.Convolution(
            data = quanData, weight = quanWeight, bias = quanBias,
            kernel = Shape(3, 3), num_filter=filters(i),
            pad = Shape(1, 1), no_bias = false, name = s"conv${i + 1}_${j + 1}")
        
        val deQuanConv = Symbol.api.broadcast_div(conv, weightScale * scale)

        internelLayer = Symbol.api.Activation(deQuanConv, act_type = "relu", name = s"relu${i + 1}_${j + 1}")
      }
      internelLayer = Symbol.api.Pooling(internelLayer, name = s"pool${i + 1}",
                                         kernel = Shape(2, 2), stride = Shape(2, 2), pool_type = "max")
    }
    internelLayer
  }

  private def getClassifier(input: Symbol, numClasses: Int): Symbol = {
    val fc6 = Symbol.api.FullyConnected(data = input, num_hidden = 512, no_bias = false, flatten = true, name = "fc6")
    val relu6 = Symbol.api.Activation(fc6, act_type = "relu", name = "relu6")
    val drop6 = Symbol.api.Dropout(data = relu6, p = 0.5f, name = "drop6")  
    val fc8 = Symbol.api.FullyConnected(data = drop6, num_hidden = numClasses, no_bias = false, name = "fc8")
    fc8
  }

  def getQuanVgg16Symbol(numClasses: Int, quantizeLevel: Int = 128): Symbol = {
    val layers = Array(2, 2, 3, 3, 3)
    val filters = Array(64, 128, 256, 512, 512)
    
    val data = Symbol.Variable("data", dType = DType.Float32)
    val feature = getQuanFeature(data, layers, filters, quantizeLevel)
    val classifier = getClassifier(feature, numClasses)
    val softmax = Symbol.api.SoftmaxActivation(data = classifier, name = "softmax")
    softmax
  }
}