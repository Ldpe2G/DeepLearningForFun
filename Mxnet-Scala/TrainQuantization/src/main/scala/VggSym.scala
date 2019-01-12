import org.apache.mxnet._
import org.apache.mxnet.util.OptionConversion._

object VggSym {

  private def getFeature(input: Symbol, layers: Array[Int], filters: Array[Int]): Symbol = {
    var internelLayer = input
    
    for (i <- 0 until layers.length) {
      for (j <- 0 until layers(i)) {
        internelLayer = Symbol.api.Convolution(
            data = internelLayer, kernel = Shape(3, 3), num_filter=filters(i),
            pad = Shape(1, 1), no_bias = false, name = s"conv${i + 1}_${j + 1}")
        internelLayer = Symbol.api.BatchNorm(data = internelLayer, name = s"bn${i + 1}_${j + 1}")
        internelLayer = Symbol.api.Activation(internelLayer, act_type = "relu", name = s"relu${i + 1}_${j + 1}")
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

  def getVgg16Symbol(numClasses: Int): Symbol = {
    val layers = Array(2, 2, 3, 3, 3)
    val filters = Array(64, 128, 256, 512, 512)
    
    val data = Symbol.Variable("data", dType = DType.Float32)
    val feature = getFeature(data, layers, filters)
    val classifier = getClassifier(feature, numClasses)
    val softmax = Symbol.api.SoftmaxOutput(data = classifier, name = "softmax")
    softmax
  }

}