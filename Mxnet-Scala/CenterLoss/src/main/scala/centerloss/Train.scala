package centerloss

import org.kohsuke.args4j.{CmdLineParser, Option}
import scala.collection.JavaConverters._
import ml.dmlc.mxnet.Operator
import ml.dmlc.mxnet.Context
import ml.dmlc.mxnet.Symbol
import ml.dmlc.mxnet.Shape
import scala.collection.immutable.ListMap
import ml.dmlc.mxnet.Mixed
import ml.dmlc.mxnet.Normal
import ml.dmlc.mxnet.Xavier
import ml.dmlc.mxnet.NDArray
import ml.dmlc.mxnet.optimizer.SGD
import ml.dmlc.mxnet.Accuracy
import ml.dmlc.mxnet.Callback.Speedometer

/**
 * @author Depeng Liang
 */
object Train {
  Operator.register("centerloss", new CenterLossProp)

  def main(args: Array[String]): Unit = {
    val leop = new Train
    val parser: CmdLineParser = new CmdLineParser(leop)
    try {
      parser.parseArgument(args.toList.asJava)
      assert(leop.dataPath != null)

      val ctx = if (leop.gpu >= 0) Context.gpu(0) else Context.cpu()
      val useCenterLoss = if (leop.withCenterLoss == 1) true else false

      val data = Symbol.Variable("data")
      val label = Symbol.Variable("label")
      val fc1 = Symbol.FullyConnected("fc1")()(Map("data" -> data, "num_hidden" -> 128))
      val act1 = Symbol.Activation("relu1")()(Map("data" -> fc1, "act_type" -> "relu"))
      val fc2 = Symbol.FullyConnected("fc2")()(Map("data" -> act1, "num_hidden" -> 64))
      val act2 = Symbol.Activation("relu2")()(Map("data" -> fc2, "act_type" -> "relu"))
      
      val fc3 = Symbol.FullyConnected("fc3")()(Map("data" -> act2, "num_hidden" -> 10))
      val mlp = Symbol.SoftmaxOutput("softmax")()(Map("data" -> fc3, "label" -> label))

      val net = if (useCenterLoss) {
        val fc4 = Symbol.FullyConnected("fc4")()(Map("data" -> act2, "num_hidden" -> 4))
        val centerLabel = Symbol.Variable("center_label")
        val centerLoss = Symbol.Custom("center_loss")()(Map("data" -> fc4, "label" -> centerLabel,
          "op_type" -> "centerloss", "num_class" -> 10, "alpha" -> 0.5f, "scale" -> 1.0f))
          
        Symbol.Group(mlp, centerLoss)
      } else mlp

      val (trainIter, testIter) =
        Data.mnistIterator(leop.dataPath, batchSize = leop.batchSize, inputShape = Shape(784))

      val datasAndLabels = 
        if (useCenterLoss) trainIter.provideData ++ trainIter.provideLabel ++ ListMap("center_label" -> trainIter.provideLabel("label"))
        else trainIter.provideData ++ trainIter.provideLabel
        
      val (argShapes, outputShapes, auxShapes) = net.inferShape(datasAndLabels)

      val initializer = new Xavier(factorType = "in", magnitude = 2.34f)
      val argNames = net.listArguments()
      val argDict = argNames.zip(argShapes.map(NDArray.empty(_, ctx))).toMap
      val auxDict = net.listAuxiliaryStates().zip(auxShapes.map(NDArray.zeros(_, ctx))).toMap

      val gradDict = argNames.zip(argShapes).filter { case (name, shape) =>
        !datasAndLabels.contains(name)
      }.map(x => x._1 -> NDArray.empty(x._2, ctx) ).toMap

      argDict.foreach { case (name, ndArray) =>
        if (!datasAndLabels.contains(name)) {
          initializer.initWeight(name, ndArray)
        }
      }

      val executor = net.bind(ctx, argDict, gradDict, "write", auxDict, null, null)
      val opt = new SGD(learningRate = leop.lr, wd = 0.0005f)
      val paramsGrads = gradDict.toList.zipWithIndex.map { case ((name, grad), idx) =>
        (idx, name, grad, opt.createState(idx, argDict(name)))
      }

      val evalMetric = new Accuracy
      val batchEndCallback = new Speedometer(100, 100)
      val numEpoch = 20

      for (epoch <- 0 until numEpoch) {
        val tic = System.currentTimeMillis
        evalMetric.reset()
        var nBatch = 0
        var epochDone = false

        trainIter.reset()
        while (!epochDone) {
          var doReset = true
          while (doReset && trainIter.hasNext) {
            val dataBatch = trainIter.next()
            argDict("data").set(dataBatch.data(0))
            argDict("label").set(dataBatch.label(0))
            if (useCenterLoss) argDict("center_label").set(dataBatch.label(0))
            executor.forward(isTrain = true)
            executor.backward()
            paramsGrads.foreach { case (idx, name, grad, optimState) =>
              opt.update(idx, argDict(name), grad, optimState)
            }
            evalMetric.update(dataBatch.label, executor.outputs.take(1))
            nBatch += 1
            batchEndCallback.invoke(epoch, nBatch, evalMetric)
          }
          if (doReset) {
            trainIter.reset()
          }
          epochDone = true
        }
        println(s"Epoch[$epoch] Train-accuracy=${evalMetric.get}")
        val toc = System.currentTimeMillis
        println(s"Epoch[$epoch] Time cost=${toc - tic}")

        evalMetric.reset()
        testIter.reset()
        while (testIter.hasNext) {
          val evalBatch = testIter.next()
          argDict("data").set(evalBatch.data(0))
          argDict("label").set(evalBatch.label(0))
          if (useCenterLoss) argDict("center_label").set(evalBatch.label(0))
          executor.forward(isTrain = false)
          evalMetric.update(evalBatch.label, executor.outputs.take(1))
          evalBatch.dispose()
        }
        println(s"Epoch[$epoch] Validation-accuracy=${evalMetric.get}")
      }
      executor.dispose()
    } catch {
      case ex: Exception => {
        println(ex.getMessage, ex)
        parser.printUsage(System.err)
        sys.exit(1)
      }
    }
  }
}

class Train {
  @Option(name = "--batch-size", usage = "the training batch size, default 100")
  private val batchSize: Int = 100
  @Option(name = "--data-path", usage = "the mnist data path")
  private val dataPath: String = null
  @Option(name = "--gpu", usage = "which gpu card to use, default is -1, means using cpu")
  private val gpu: Int = -1
  @Option(name = "--lr", usage = "init learning rate")
  private val lr: Float = 0.001f
  @Option(name = "--with-center-loss", usage = "whether train with center loss, default 1 means use center loss, set to 0 if not")
  private val withCenterLoss: Int = 1
}
