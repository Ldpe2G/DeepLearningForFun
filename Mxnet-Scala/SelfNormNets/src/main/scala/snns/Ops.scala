package snns

import org.apache.mxnet.Symbol
import org.apache.mxnet.Shape
import org.apache.mxnet.SymbolConversions._
import org.apache.mxnet.util.OptionConversion._

object Ops {

  def selu(x: Symbol): Symbol = {
    val alpha = 1.6732632423543772848170429916717f
    val scale = 1.0507009873554804934193349852946f
    val condition = x >= 0f
    val y = Symbol.api.LeakyReLU(x, act_type = "elu", slope = alpha)
    scale * Symbol.api.where(condition = condition, x = x, y = y)
  }

  def dropoutSelu(x: Symbol, rate: Float, xShape: Shape = null,
      alpha: Float = -1.7580993408473766f, fixedPointMean: Float = 0f, fixedPointVar: Float =1f): Symbol = {
    val keepProp = 1f - rate
    require(keepProp <= 1f && keepProp > 0f,
        s"keepProb must be a scalar tensor or a float in the range (0, 1], got $keepProp")
    if (keepProp == 1f) x
    else {
      require(xShape != null)
      var randomTensor = Symbol.random.uniform(low=0, high=1, shape = xShape)
      randomTensor = randomTensor + keepProp
      var binaryTensor = Symbol.api.floor(randomTensor)
      binaryTensor = Symbol.api.BlockGrad(binaryTensor)
      var ret = x * binaryTensor + alpha * (1 - binaryTensor)
      val a = Math.sqrt(fixedPointVar / (keepProp * ((1 - keepProp) * Math.pow(alpha, 2) + fixedPointVar))).toFloat
      val b = fixedPointMean - a * (keepProp * fixedPointMean + (1 - keepProp) * alpha)
      a * ret + b
    }
  }
}
