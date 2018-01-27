package example

import mxgan.Viz._
import ml.dmlc.mxnet.Context
import ml.dmlc.mxnet.Shape
import ml.dmlc.mxnet.IO
import ml.dmlc.mxnet.NDArray
import ml.dmlc.mxnet.CustomMetric
import mxgan.Generator
import mxgan.Encoder
import mxgan.GANModule
import ml.dmlc.mxnet.Xavier
import ml.dmlc.mxnet.optimizer.Adam
import ml.dmlc.mxnet.DataBatch
import mxgan.Viz

object GanMnist {

  // Evaluation 
  def ferr(label: NDArray, pred: NDArray): Float = {
    val predArr = pred.toArray.map(p => if (p > 0.5) 1f else 0f)
    val labelArr = label.toArray
    labelArr.zip(predArr).map { case (l, p) => Math.abs(l - p) }.sum / label.shape(0)
  }

  def main(args: Array[String]): Unit = {
    assert(args.length == 2, "usage: GanMnist mnist-data-path gpu_id")
    
    val dataDir = args(0)
    val gpu = args(1).toInt
      
    val lr = 0.0005f
    val beta1 = 0.5f
    val batchSize = 100
    val randShape = Shape(batchSize, 100)
    val numEpoch = 100
    val dataShape = Shape(batchSize, 1, 28, 28)
    val context = if (gpu == -1) Context.cpu() else Context.gpu(gpu)
    
    val symGen = Generator.dcgan28x28(oShape = dataShape, ngf = 32, finalAct = "sigmoid")
    val symDec = Encoder.lenet()
    
    val gMod = new GANModule(
        symGen,
        symDec,
        context = context,
        dataShape = dataShape,
        codeShape = randShape)

    gMod.initGParams(new Xavier(factorType = "in", magnitude = 2.34f))
    gMod.initDParams(new Xavier(factorType = "in", magnitude = 2.34f))

    gMod.initOptimizer(new Adam(learningRate = lr, wd = 0f, beta1 = beta1))
    
    val params = Map(
      "image" -> s"$dataDir/train-images-idx3-ubyte",
      "label" -> s"$dataDir/train-labels-idx1-ubyte",
      "input_shape" -> s"(1, 28, 28)",
      "batch_size" -> s"$batchSize",
      "shuffle" -> "True"
    )

    val mnistIter = IO.MNISTIter(params)
    
    val metricAcc = new CustomMetric(ferr, "ferr")
    
    var t = 0
    var dataBatch: DataBatch = null
    for (epoch <- 0 until numEpoch) {
      mnistIter.reset()
      metricAcc.reset()
      t = 0
      while (mnistIter.hasNext) {
        dataBatch = mnistIter.next()
        gMod.update(dataBatch)
        gMod.dLabel.set(0f)
        metricAcc.update(Array(gMod.dLabel), gMod.outputsFake)
        gMod.dLabel.set(1f)
        metricAcc.update(Array(gMod.dLabel), gMod.outputsReal)
        
        if (t % 50 == 0) {
          val (name, value) = metricAcc.get
          println(s"epoch: $epoch, iter $t, metric=${value(0)}")
          Viz.imshow("gout", gMod.tempOutG(0), 2, flip = true)
          val diff = gMod.tempDiffD
          val arr = diff.toArray
          val mean = arr.sum / arr.length
          val std = {
            val tmpA = arr.map(a => (a - mean) * (a - mean))
            Math.sqrt(tmpA.sum / tmpA.length).toFloat
          }
          diff.set((diff - mean) / std + 0.5f)
          Viz.imshow("diff", diff, flip = true)
          Viz.imshow("data", dataBatch.data(0), flip = true)
        }

        t += 1
      }
    }
  }

}
