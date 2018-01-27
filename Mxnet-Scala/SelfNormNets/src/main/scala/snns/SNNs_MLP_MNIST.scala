package snns

import org.kohsuke.args4j.{CmdLineParser, Option}
import scala.collection.JavaConverters._
import ml.dmlc.mxnet.Context
import ml.dmlc.mxnet.IO
import ml.dmlc.mxnet.Symbol
import ml.dmlc.mxnet.Shape
import ml.dmlc.mxnet.Random
import ml.dmlc.mxnet.optimizer.SGD
import ml.dmlc.mxnet.Accuracy
import ml.dmlc.mxnet.Callback.Speedometer
import ml.dmlc.mxnet.optimizer.RMSProp

/**
 * @author Depeng Liang
 */
object SNNs_MLP_MNIST {

  // Create model
  def multilayerPerceptron(batchSize: Int, rate: Float, nClass: Int, nHidden1: Int, nHidden2: Int): Symbol = {
    val data = Symbol.Variable("data")
    val label = Symbol.Variable("label")
    
    var layer1 = Symbol.FullyConnected(name = "fc1")()(Map("data" -> data, "num_hidden" -> nHidden1))
    layer1 = Ops.selu(layer1)
    val layer1Shape = Shape(batchSize, nHidden1)
    layer1 = Ops.dropoutSelu(layer1, rate, layer1Shape)

    var layer2 = Symbol.FullyConnected(name = "fc2")()(Map("data" -> layer1, "num_hidden" -> nHidden2))
    layer2 = Ops.selu(layer2)
    val layer2Shape = Shape(batchSize, nHidden2)
    layer2 = Ops.dropoutSelu(layer2, rate, layer2Shape)
    
    val out = Symbol.FullyConnected(name = "out")()(Map("data" -> layer2, "num_hidden" -> nClass))
    Symbol.SoftmaxOutput("softmax")()(Map("data" -> out, "label" -> label))
  }

  def main(args: Array[String]): Unit = {
    val snst = new SNNs_MLP_MNIST
    val parser: CmdLineParser = new CmdLineParser(snst)
    try {
      parser.parseArgument(args.toList.asJava)
      assert(snst.dataPath != null)

      val ctx = if (snst.gpu >= 0) Context.gpu(snst.gpu) else Context.cpu()

      // Network Parameters
      val nHidden1 = 784 // 1st layer number of features
      val nHidden2 = 784 // 2nd layer number of features
      val nInput = 784 // MNIST data input (img shape: 28*28)
      val nClasses = 10 // MNIST total classes (0-9 digits)

      val trainIter = IO.MNISTIter(Map(
        "image" -> s"${snst.dataPath}/train-images-idx3-ubyte",
        "label" -> s"${snst.dataPath}/train-labels-idx1-ubyte",
        "label_name" -> "softmax_label",
        "input_shape" -> s"($nInput,)",
        "batch_size" -> snst.batchSize.toString,
        "shuffle" -> "True",
        "flat" -> "True", "silent" -> "False", "seed" -> "10"))

      val testIter = IO.MNISTIter(Map(
        "image" -> s"${snst.dataPath}/t10k-images-idx3-ubyte",
        "label" -> s"${snst.dataPath}/t10k-labels-idx1-ubyte",
        "label_name" -> "softmax_label",
        "input_shape" -> s"($nInput,)",
        "batch_size" -> snst.batchSize.toString,
        "flat" -> "True", "silent" -> "False"))

      val trainNet = multilayerPerceptron(snst.batchSize, 0.05f, nClasses, nHidden1, nHidden2)
      val predNet = multilayerPerceptron(snst.batchSize, 0f, nClasses, nHidden1, nHidden2)

      val execTrain = trainNet.simpleBind(ctx, gradReq = "write", shapeDict = Map("data" -> Shape(snst.batchSize, nInput)))
      val execPred = predNet.simpleBind(ctx, gradReq = "write", shapeDict = Map("data" -> Shape(snst.batchSize, nInput)))

      Random.normal(loc = 0f, scale = Math.sqrt(1.0 / nInput).toFloat, out = execTrain.argDict("fc1_weight"))
      Random.normal(loc = 0f, scale = Math.sqrt(1.0 / nHidden1).toFloat, out = execTrain.argDict("fc2_weight")) 
      Random.normal(loc = 0f, scale = Math.sqrt(1.0 / nHidden2).toFloat, out = execTrain.argDict("out_weight")) 
      Random.normal(loc = 0f, scale = 1e-6f, out = execTrain.argDict("fc1_bias"))
      Random.normal(loc = 0f, scale = 1e-6f, out = execTrain.argDict("fc2_bias"))
      Random.normal(loc = 0f, scale = 1e-6f, out = execTrain.argDict("out_bias"))

      val opt = new RMSProp(learningRate = snst.lr)
      val paramsGrads = execTrain.gradDict.filter{ case (n, d) => n != "data" && n != "label" }
                                                          .toList.zipWithIndex.map { case ((name, grad), idx) =>
        (idx, name, grad, opt.createState(idx, execTrain.argDict(name)))
      }

      val evalMetric = new Accuracy
      val batchEndCallback = new Speedometer(100, 100)

      for (epoch <- 0 until snst.trainEpoch) {
        val tic = System.currentTimeMillis
        evalMetric.reset()
        var nBatch = 0
        var epochDone = false

        trainIter.reset()
        while (!epochDone) {
          var doReset = true
          while (doReset && trainIter.hasNext) {
            val dataBatch = trainIter.next()
            execTrain.argDict("data").set(dataBatch.data(0))
            execTrain.argDict("label").set(dataBatch.label(0))

            execTrain.forward(isTrain = true)
            execTrain.backward()

            paramsGrads.foreach { case (idx, name, grad, optimState) =>
              opt.update(idx, execTrain.argDict(name), grad, optimState)
            }
            evalMetric.update(dataBatch.label, execTrain.outputs.take(1))
            nBatch += 1
            batchEndCallback.invoke(epoch, nBatch, evalMetric)
          }
          if (doReset) {
            trainIter.reset()
          }
          epochDone = true
        }
        val (name, value) = evalMetric.get
        name.zip(value).foreach { case (n, v) =>
          println(s"Epoch[$epoch] Train-accuracy=$v")
        }
        val toc = System.currentTimeMillis
        println(s"Epoch[$epoch] Time cost=${toc - tic}")

        execPred.argDict.foreach { case (name, nd) =>
          if (execTrain.argDict.contains(name)) nd.set(execTrain.argDict(name))
        }
        evalMetric.reset()
        testIter.reset()
        while (testIter.hasNext) {
          val evalBatch = testIter.next()
          
          execPred.argDict("data").set(evalBatch.data(0))
          execPred.argDict("label").set(evalBatch.label(0))
          
          execPred.forward(isTrain = false)
          evalMetric.update(evalBatch.label, execPred.outputs.take(1))
          evalBatch.dispose()
        }
        val (names, values) = evalMetric.get
        names.zip(values).foreach { case (n, v) =>
          println(s"Epoch[$epoch] Validation-accuracy=$v")
        }
      }
      execTrain.dispose()
      execPred.dispose()
      
    } catch {
      case ex: Exception => {
        println(ex.getMessage, ex)
        parser.printUsage(System.err)
        sys.exit(1)
      }
    }
  }
}

class SNNs_MLP_MNIST {
  @Option(name = "--batch-size", usage = "the training batch size, default 100")
  private var batchSize: Int = 100
  @Option(name = "--data-path", usage = "the mnist data path")
  private var dataPath: String = null
  @Option(name = "--gpu", usage = "which gpu card to use, default is -1, means using cpu")
  private var gpu: Int = -1
  @Option(name = "--lr", usage = "init learning rate")
  private var lr: Float = 0.001f
  @Option(name = "--train-epoch", usage = "training epoch")
  private var trainEpoch: Int = 15
}
