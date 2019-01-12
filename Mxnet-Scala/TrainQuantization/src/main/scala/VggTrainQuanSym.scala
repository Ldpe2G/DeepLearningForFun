import org.apache.mxnet._
import org.apache.mxnet.util.OptionConversion._
import QuantizeOps._

object VggQuanSym {
  private def getQuanFeature(input: Symbol, layers: Array[Int], filters: Array[Int]): Symbol = {
    var internelLayer = input
    
    for (i <- 0 until layers.length) {
      for (j <- 0 until layers(i)) {
        val scale = Symbol.Variable(s"conv${i + 1}_${j + 1}_in_act_scale_beta")
        val movingMax = Symbol.Variable(s"conv${i + 1}_${j + 1}_in_act_moving_max_beta")
        val actKwargs = scala.collection.mutable.Map[String, Any](
          "data" -> internelLayer, "scale" -> scale, "moving_max" -> movingMax, 
          "quantize_weight" -> 0, "quantize_level" -> 128, "momentum" -> 0.95f)
        Operator.register(s"conv${i + 1}_${j + 1}_in_act_quantizer", new TrainZeroCenteredQuantizerProp())
        val actQuantizer = Symbol.api.Custom(s"conv${i + 1}_${j + 1}_in_act_quantizer", 
            name = s"conv${i + 1}_${j + 1}_in_act_quantizer", kwargs = actKwargs)
            
        val weight = Symbol.Variable(s"conv${i + 1}_${j + 1}_weight")
        val bias = Symbol.Variable(s"conv${i + 1}_${j + 1}_bias")
        var fakeConv = Symbol.api.Convolution(
            data = internelLayer, weight = weight, bias = bias, 
            kernel = Shape(3, 3), num_filter=filters(i),
            pad = Shape(1, 1), no_bias = false, name = s"fake_conv${i + 1}_${j + 1}")
        fakeConv = Symbol.api.BlockGrad(fakeConv)
        
        val gamma = Symbol.Variable(s"bn${i + 1}_${j + 1}_gamma")
        val beta = Symbol.Variable(s"bn${i + 1}_${j + 1}_beta")
        val moving_mean = Symbol.Variable(s"bn${i + 1}_${j + 1}_moving_mean")
        val moving_var = Symbol.Variable(s"bn${i + 1}_${j + 1}_moving_var")
        val bn = Symbol.api.BatchNorm(
            fakeConv, gamma, beta, moving_mean, moving_var, 
            fix_gamma = true, use_global_stats = true,
            output_mean_var = false, name = s"bn${i + 1}_${j + 1}")
        val bnOut = Symbol.api.BlockGrad(bn)
        
        Operator.register(s"merge_bn${i + 1}_${j + 1}", new MergeBNProp())
        val bnWargs = scala.collection.mutable.Map[String, Any](
            "weight" -> weight, "bias" -> bias,
            "gamma" -> Symbol.api.BlockGrad(gamma),
            "beta" -> Symbol.api.BlockGrad(beta),
            "moving_mean" -> Symbol.api.BlockGrad(moving_mean),
            "moving_var" -> Symbol.api.BlockGrad(moving_var)
        )
        val mergeBn: Symbol = Symbol.api.Custom(s"merge_bn${i + 1}_${j + 1}", name = s"merge_bn${i + 1}_${j + 1}", kwargs = bnWargs)

        val weightScale = Symbol.Variable(s"conv${i + 1}_${j + 1}_weight_scale_beta")
        val weightKwargs = scala.collection.mutable.Map[String, Any](
          "data" -> mergeBn.get(0), "scale" -> weightScale,
          "quantize_weight" -> 1, "quantize_level" -> 128)
        Operator.register(s"conv${i + 1}_${j + 1}_weight_quantizer", new TrainZeroCenteredQuantizerProp())
        val weightQuantizer = Symbol.api.Custom(s"conv${i + 1}_${j + 1}_weight_quantizer", 
            name = s"conv${i + 1}_${j + 1}_weight_quantizer", kwargs = weightKwargs)

        val conv = Symbol.api.Convolution(
            data = actQuantizer, weight = weightQuantizer, bias = mergeBn.get(1),
            kernel = Shape(3, 3), num_filter=filters(i),
            pad = Shape(1, 1), no_bias = false, name = s"conv${i + 1}_${j + 1}")
            
        Operator.register(s"conv${i + 1}_${j + 1}_merge_two_streams", new MergeTwoStreamsProp())
        val merge = Symbol.api.Custom(
            s"conv${i + 1}_${j + 1}_merge_two_streams",
            name = s"conv${i + 1}_${j + 1}_merge_two_streams",
          kwargs = scala.collection.mutable.Map[String, Any](
            "data_left" -> bnOut, "data_right" -> conv, "index" -> 1    
          ))
        internelLayer = Symbol.api.Activation(merge, act_type = "relu", name = s"relu${i + 1}_${j + 1}")
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

  def getQuanVgg16Symbol(numClasses: Int): Symbol = {
    val layers = Array(2, 2, 3, 3, 3)
    val filters = Array(64, 128, 256, 512, 512)
    
    val data = Symbol.Variable("data", dType = DType.Float32)
    val feature = getQuanFeature(data, layers, filters)
    val classifier = getClassifier(feature, numClasses)
    val softmax = Symbol.api.SoftmaxOutput(data = classifier, name = "softmax")
    softmax
  }
}