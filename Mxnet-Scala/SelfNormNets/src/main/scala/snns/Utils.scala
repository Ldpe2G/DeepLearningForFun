package snns

import ml.dmlc.mxnet.NDArray

object Utils {
  def getSoftMaxLoss(pred: NDArray, label: NDArray): Float = {
    val shape = pred.shape
    val maxIdx = NDArray.argmax_channel(pred).toArray
    val loss = pred.toArray.grouped(shape(1)).zipWithIndex.map { case (array, idx) =>
        array(maxIdx(idx).toInt)  
      }.map(-Math.log(_)).sum.toFloat
    loss
  }
}