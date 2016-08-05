package example

import mxgan.Viz._
import ml.dmlc.mxnet.Shape
import ml.dmlc.mxnet.Context
import ml.dmlc.mxnet.IO
import ml.dmlc.mxnet.NDArray
import ml.dmlc.mxnet.CustomMetric
import mxgan.Generator
import mxgan.Encoder
import ml.dmlc.mxnet.DataBatch
import mxgan.GANModule
import ml.dmlc.mxnet.Normal
import ml.dmlc.mxnet.Xavier
import ml.dmlc.mxnet.optimizer.Adam
import mxgan.Viz

object GanCifar10 {
  
  // Evaluation 
  def ferr(label: NDArray, pred: NDArray): Float = {
    val predArr = pred.toArray.map(p => if (p > 0.5) 1f else 0f)
    val labelArr = label.toArray
    labelArr.zip(predArr).map { case (l, p) => Math.abs(l - p) }.sum / label.shape(0)
  }

  def main(args: Array[String]): Unit = {
    assert(args.length == 2, "usage: GanCifar10 cifar-data-path gpu_id")
    
    val dataPath = args(0)
    val gpu = args(1).toInt
    
    val ngf = 64
    val lr = 0.0003f
    val beta1 = 0.5f
    val batchSize = 100
    val randShape = Shape(batchSize, 100)
    val numEpoch = 100
    val dataShape = Shape(batchSize, 3, 32, 32)
    val context = if (gpu == -1) Context.cpu() else Context.gpu(gpu)

    val symGen = Generator.dcgan32x32(oShape = dataShape, ngf = ngf, finalAct = "tanh")
    val symDec = Encoder.dcgan(ngf = ngf / 2)

    val gMod = new GANModule(
        symGen,
        symDec,
        context = context,
        dataShape = dataShape,
        codeShape = randShape)

    gMod.initGParams(new Normal(0.05f))
    gMod.initDParams(new Xavier(factorType = "in", magnitude = 2.34f))

    gMod.initOptimizer(new Adam(learningRate = lr, wd = 0f, beta1 = beta1))

    val params = Map(
      "path_imgrec" -> s"${dataPath}/train.rec",
      "shuffle" -> "True",
      "data_shape" -> "(3, 32, 32)",
      "batch_size" -> s"$batchSize"
    )
    val imgRecIter = IO.ImageRecordIter(params)
    
    val metricAcc = new CustomMetric(ferr, "ferr")

    var t = 0
    var dataBatch: DataBatch = null
    for (epoch <- 0 until numEpoch) {
      imgRecIter.reset()
      metricAcc.reset()
      t = 0
      while (imgRecIter.hasNext) {
        dataBatch = imgRecIter.next()
        val realImgs = dataBatch.data(0).copyTo(context)
        realImgs.set(realImgs * (1.0f / 255.0f) - 0.5f)
        gMod.update(new DataBatch(Array(realImgs), dataBatch.label, dataBatch.index, dataBatch.pad))
        gMod.dLabel.set(0f)
        metricAcc.update(Array(gMod.dLabel), gMod.outputsFake)
        gMod.dLabel.set(1f)
        metricAcc.update(Array(gMod.dLabel), gMod.outputsReal)
        
        if (t % 50 == 0) {
          val (name, value) = metricAcc.get
          println(s"epoch: $epoch, iter $t, metric=$value")
          Viz.imshow("gout", gMod.tempOutG(0) + 0.5f, 2, flip = true)
          val diff = gMod.tempDiffD
          val arr = diff.toArray
          val mean = arr.sum / arr.length
          val std = {
            val tmpA = arr.map(a => (a - mean) * (a - mean))
            Math.sqrt(tmpA.sum / tmpA.length).toFloat
          }
          diff.set((diff - mean) / std + 0.5f)
          Viz.imshow("diff", diff, flip = true)
          Viz.imshow("data", realImgs + 0.5f, flip = true)
        }

        t += 1
      }
    }
  }
}
