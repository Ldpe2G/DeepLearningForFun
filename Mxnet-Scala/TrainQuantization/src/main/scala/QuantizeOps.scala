

import org.apache.mxnet.CustomOp
import org.apache.mxnet.CustomOpProp
import org.apache.mxnet.Shape
import org.apache.mxnet.NDArray
import org.apache.mxnet.DType.DType
import org.apache.mxnet.NDArrayCollector
import org.apache.mxnet.Operator

/**
 * @author Depeng Liang
 */
object QuantizeOps {
  class TrainZeroCenteredQuantizer(ctx: String, inShapes: Array[Array[Int]], inDtypes: Array[Int],
      quantizeWeight: Int = 1, quantizeLevel: Int = 128, momentum: Float = 0.95f) extends CustomOp {
  
    override def forward(isTrain: Boolean, req: Array[String],
        inData: Array[NDArray], outData: Array[NDArray], aux: Array[NDArray]): Unit = {
      NDArrayCollector.auto().withScope {
        if (this.quantizeWeight == 1) {
          val scale = aux(0)
          val abs = NDArray.api.abs(inData(0))
          val max = NDArray.api.max(abs).get.toScalar
          val s = (this.quantizeLevel - 1).toFloat / max
          scale.set(s)
          val outND = NDArray.api.clip(NDArray.api.round(inData(0) * s), -(this.quantizeLevel - 1).toFloat, this.quantizeLevel.toFloat - 1).get / s
          this.assign(outData(0), req(0), outND)
        } else {  // quantize activation
          val scale = aux(0)
          val movingMax = aux(1)
          val abs = NDArray.api.abs(inData(0))
          val max = NDArray.api.max(abs).get.toScalar
          
          val sum = movingMax.toScalar
          if (sum == 0f) {
            movingMax.set(max)
          }
          
          if (isTrain == true) {
            scale.set(this.quantizeLevel / movingMax.toScalar)
          }
          
          val s = scale.toScalar
          val outND = NDArray.api.clip(NDArray.api.round(inData(0) * s), -this.quantizeLevel.toFloat, this.quantizeLevel.toFloat - 1).get / s
          this.assign(outData(0), req(0), outND)
        }
      }
    }
  
    override def backward(req: Array[String], outGrad: Array[NDArray],
      inData: Array[NDArray], outData: Array[NDArray],
      inGrad: Array[NDArray], aux: Array[NDArray]): Unit = {
        NDArrayCollector.auto().withScope {
          if (this.quantizeWeight == 0) {
            val scale = aux(0)
            val movingMax = aux(1)
            
            val abs = NDArray.api.abs(inData(0))
            val max = NDArray.api.max(abs).get.toScalar
            
            val sum = movingMax.toScalar
            if (sum == 0f) {
              movingMax.set(max)
            } else {
              movingMax.set(this.momentum * sum + (1 - this.momentum) * max)
            }
          }
          this.assign(inGrad(0), req(0), outGrad(0))
        }
    }
  }

  class TrainZeroCenteredQuantizerProp(needTopGrad: Boolean = true)
      extends CustomOpProp(needTopGrad) {
  
    override def listArguments(): Array[String] = Array("data")
  
    override def listOutputs(): Array[String] = Array("output")
  
    override def listAuxiliaryStates(): Array[String] = {
      require(this.kwargs.contains("quantize_weight"))
      val quantizeWeight = this.kwargs("quantize_weight").toInt
      if (quantizeWeight == 1) {
        Array("scale")
      } else {
        Array("scale", "moving_max")
      }
    }

    override def inferShape(inShape: Array[Shape]):
        (Array[Shape], Array[Shape], Array[Shape]) = {
      require(this.kwargs.contains("quantize_weight"))
      val quantizeWeight = this.kwargs("quantize_weight").toInt
      
      val dataShape = inShape(0)
      
      val scaleShape = Shape(1)
      val movingMaxShape = Shape(1)
      
      var auxShapes = Array(scaleShape)
      
      if (quantizeWeight == 0) {
        auxShapes = auxShapes :+ movingMaxShape
      }
      
      (Array(dataShape), Array(dataShape), auxShapes)
    }
  
    override def inferType(inType: Array[DType]):
      (Array[DType], Array[DType], Array[DType]) = {
      require(this.kwargs.contains("quantize_weight"))
      val quantizeWeight = this.kwargs("quantize_weight").toInt
      
      if (quantizeWeight == 1) {
        (inType, inType.take(1), Array.fill[DType](1)(inType(0)))
      } else {
        (inType, inType.take(1), Array.fill[DType](2)(inType(0)))
      }
    }
  
    override def createOperator(ctx: String, inShapes: Array[Array[Int]],
      inDtypes: Array[Int]): CustomOp = {
      require(this.kwargs.contains("quantize_weight") 
          && this.kwargs.contains("quantize_level"))
      val quantizeWeight = this.kwargs("quantize_weight").toInt
      val quantizeLevel = this.kwargs("quantize_level").toInt
      val momentum = if (this.kwargs.contains("momentum")) this.kwargs("momentum").toFloat else 0.95f
      new TrainZeroCenteredQuantizer(ctx, inShapes, inDtypes, quantizeWeight, quantizeLevel, momentum)
    }
  }
  
  class MergeTwoStreams(ctx: String, inShapes: Array[Array[Int]], inDtypes: Array[Int],
      index: Int = 1) extends CustomOp {
  
    override def forward(isTrain: Boolean, req: Array[String],
        inData: Array[NDArray], outData: Array[NDArray], aux: Array[NDArray]): Unit = {
      this.assign(outData(0), req(0), inData(this.index))
    }
  
    override def backward(req: Array[String], outGrad: Array[NDArray],
      inData: Array[NDArray], outData: Array[NDArray],
      inGrad: Array[NDArray], aux: Array[NDArray]): Unit = {
        this.assign(inGrad(0), req(0), outGrad(0))
        this.assign(inGrad(1), req(1), outGrad(0))
    }
  }
  
  class MergeTwoStreamsProp(needTopGrad: Boolean = true)
      extends CustomOpProp(needTopGrad) {
  
    override def listArguments(): Array[String] = Array("data_left", "data_right")
  
    override def listOutputs(): Array[String] = Array("output")

    override def inferShape(inShape: Array[Shape]):
        (Array[Shape], Array[Shape], Array[Shape]) = {
      assert(inShape(0) == inShape(1))
      
      (inShape, Array(inShape(0)), null)
    }

    override def inferType(inType: Array[DType]):
      (Array[DType], Array[DType], Array[DType]) = {
      (inType, inType.take(1), null)
    }
  
    override def createOperator(ctx: String, inShapes: Array[Array[Int]],
      inDtypes: Array[Int]): CustomOp = {
      require(this.kwargs.contains("index"), "need to provide index.")
      val index = this.kwargs("index").toInt
      new MergeTwoStreams(ctx, inShapes, inDtypes, index)
    }
  }
  
  class MergeBN(ctx: String, inShapes: Array[Array[Int]], inDtypes: Array[Int],
      eps: Float = 0.001f) extends CustomOp {
  
    override def forward(isTrain: Boolean, req: Array[String],
        inData: Array[NDArray], outData: Array[NDArray], aux: Array[NDArray]): Unit = {
      NDArrayCollector.auto().withScope {
        val weight = inData(0)
        val bias = inData(1)
        val gamma = inData(2)
        val beta = inData(3)
        val movingMean = inData(4)
        val movingVar = inData(5)
//        val mean = inData(6)
//        val `var` = inData(7)
        
        val factor = gamma / NDArray.api.sqrt(movingVar + this.eps)
        val reshapedFactor = factor.reshape(Array(factor.shape(0)) ++ Array.fill(weight.shape.length - 1)(1))
        
        val newWeight = NDArray.api.broadcast_mul(reshapedFactor, weight)
        val newBias = beta + factor * bias - factor * movingMean
        
        this.assign(outData(0), req(0), newWeight)
        this.assign(outData(1), req(1), newBias)
      }
    }

    override def backward(req: Array[String], outGrad: Array[NDArray],
      inData: Array[NDArray], outData: Array[NDArray],
      inGrad: Array[NDArray], aux: Array[NDArray]): Unit = {
        this.assign(inGrad(0), req(0), outGrad(0))
        this.assign(inGrad(1), req(1), outGrad(1))
    }
  }
  
  class MergeBNProp(needTopGrad: Boolean = true)
      extends CustomOpProp(needTopGrad) {

    override def listArguments(): Array[String] = {
      Array("weight", "bias", "gamma", "beta", "moving_mean", "moving_var")//, "mean", "var")
    }

    override def listOutputs(): Array[String] = {
      Array("weight", "bias")
    }
    
    override def inferShape(inShape: Array[Shape]):
        (Array[Shape], Array[Shape], Array[Shape]) = {
      (inShape, inShape.take(2), null)
    }

    override def inferType(inType: Array[DType]):
      (Array[DType], Array[DType], Array[DType]) = {
      (inType, inType.take(2), null)
    }

    override def createOperator(ctx: String, inShapes: Array[Array[Int]],
      inDtypes: Array[Int]): CustomOp = {
      val eps = if (this.kwargs.contains("eps")) {
        this.kwargs("eps").toFloat
      } else 0.001f
      new MergeBN(ctx, inShapes, inDtypes, eps)
    }
  }

}