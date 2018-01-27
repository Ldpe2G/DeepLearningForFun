package centerloss

import ml.dmlc.mxnet.CustomOp
import ml.dmlc.mxnet.CustomOpProp
import ml.dmlc.mxnet.Shape
import ml.dmlc.mxnet.NDArray
import ml.dmlc.mxnet.DType.DType
import scala.collection.mutable.ArrayBuffer

/**
 * Implementation of center loss
 * @author Depeng Liang
 */
class CenterLoss(ctx: String, inShapes: Array[Array[Int]], inDtypes: Array[Int],
    numClass: Int, alpha: Float, scale: Float) extends CustomOp {
  require(inShapes(0).length == 2, "dim for input_data shoudl be 2 for CenterLoss")

  private val batchSize = inShapes(0)(0)
  private val feaDim = inShapes(0)(1)

  override def forward(sTrain: Boolean, req: Array[String],
      inData: Array[NDArray], outData: Array[NDArray], aux: Array[NDArray]): Unit = {
    val datas = inData(0).toArray.grouped(this.feaDim).toArray
    val labels = inData(1).toArray
    val diff = aux(0)
    val center = aux(1).toArray.grouped(this.feaDim).toArray

    for (i <- 0 until this.batchSize) {
      diff.slice(i).set(datas(i).zip(center(labels(i).toInt)).map(x => x._1 - x._2))
    }

    val loss = NDArray.sum(NDArray.square(diff)).get / this.batchSize.toFloat / 2f
    this.assign(outData(0), req(0), loss)
    loss.disposeDepsExcept(diff)
    loss.dispose()
  }

  override def backward(req: Array[String], outGrad: Array[NDArray],
    inData: Array[NDArray], outData: Array[NDArray],
    inGrad: Array[NDArray], aux: Array[NDArray]): Unit = {
    val diff = aux(0)
    val center = aux(1)
    val sum = aux(2)

    // back grad is just scale * ( x_i - c_yi)
    val gradScale = this.scale / this.batchSize
    val scaledGrad = diff * gradScale

    this.assign(inGrad(0), req(0), scaledGrad)
    scaledGrad.dispose()

    // update the center
    val labels = inData(1).toArray
    val labelOccur = scala.collection.mutable.Map[Float, ArrayBuffer[Int]]()
    labels.zipWithIndex.foreach { case (label, idx) =>
      if (labelOccur.contains(label)) labelOccur(label) += idx
      else labelOccur += label -> ArrayBuffer(idx)
    }

    val diffArr = diff.toArray.grouped(this.feaDim).toArray
    for ((label, sampleIndex) <- labelOccur) {
      sum.set(0f)
      val sumArr = (sum.toArray /: sampleIndex.toArray){ (acc, idx) =>
        acc.zip(diffArr(idx)).map(x => x._1 + x._2)
      }
      sum.set(sumArr)
      val deltaC = sum / (1f + sampleIndex.length) * this.alpha
      val cs = center.slice(label.toInt)
      cs.set(cs.toArray.zip(deltaC.toArray).map(x => x._1 + x._2))
      deltaC.dispose()
    }
  }
}

class CenterLossProp(needTopGrad: Boolean = false)
    extends CustomOpProp(needTopGrad) {

  override def listArguments(): Array[String] = Array("data", "label")

  override def listOutputs(): Array[String] = Array("output")

  override def listAuxiliaryStates(): Array[String] = Array("diff_bias", "center_bias", "sum_bias")

  override def inferShape(inShape: Array[Shape]):
      (Array[Shape], Array[Shape], Array[Shape]) = {
    val dataShape = inShape(0)
    val labelShape = Shape(dataShape(0))

    // store diff , same shape as input batch
    val diffShape = Shape(dataShape(0), dataShape(1))

    require(this.kwargs.contains("num_class"))
    val numClass = this.kwargs("num_class").toInt
    // store the center of each class , should be ( numClass, d )
    val centerShape = Shape(numClass, diffShape(1))

    // computation buf
    val sumShape = Shape(diffShape(1))

    val outputShape = Shape(1)
    (Array(dataShape, labelShape), Array(outputShape), Array(diffShape, centerShape, sumShape))
  }

  override def inferType(inType: Array[DType]):
    (Array[DType], Array[DType], Array[DType]) = {
    (inType, inType.take(1), Array.fill[DType](3)(inType(0)))
  }

  override def createOperator(ctx: String, inShapes: Array[Array[Int]],
    inDtypes: Array[Int]): CustomOp = {
    require(this.kwargs.contains("num_class") && this.kwargs.contains("alpha"))
    val numClass = this.kwargs("num_class").toInt
    val alpha = this.kwargs("alpha").toFloat
    val scale = if (this.kwargs.contains("scale")) this.kwargs("scale").toFloat else 1.0f
    new CenterLoss(ctx, inShapes, inDtypes, numClass, alpha, scale)
  }
}
