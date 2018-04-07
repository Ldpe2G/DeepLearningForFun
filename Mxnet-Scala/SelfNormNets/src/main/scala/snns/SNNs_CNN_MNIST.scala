package snns

import org.kohsuke.args4j.{CmdLineParser, Option}
import scala.collection.JavaConverters._
import org.apache.mxnet.Context
import org.apache.mxnet.IO
import org.apache.mxnet.Symbol
import org.apache.mxnet.Shape
import org.apache.mxnet.Random
import org.apache.mxnet.optimizer.SGD
import org.apache.mxnet.Accuracy
import org.apache.mxnet.Callback.Speedometer
import org.apache.mxnet.optimizer.RMSProp
import scala.collection.mutable.ArrayBuffer

/**
 * @author Depeng Liang
 */
object SNNs_CNN_MNIST {

  // Create model
  def convNetReLU(nClass: Int): Symbol = {
    val data = Symbol.Variable("data")
    val label = Symbol.Variable("label")
    
    var conv1 = Symbol.Convolution("conv1")()(
        Map("data" -> data, "kernel" -> "(5, 5)", "pad" -> "(2, 2)", "num_filter" -> 32))
    conv1 = Symbol.Pooling("pool1")()(
        Map("data" -> conv1, "pool_type" -> "max", "kernel" -> "(2, 2)", "stride" -> "(2, 2)"))
  
    var conv2 = Symbol.Convolution("conv2")()(
        Map("data" -> conv1, "kernel" -> "(5, 5)", "pad" -> "(2, 2)", "num_filter" -> 64))
    conv2 = Symbol.Pooling("pool2")()(
        Map("data" -> conv2, "pool_type" -> "max", "kernel" -> "(2, 2)", "stride" -> "(2, 2)"))
    
    conv2 = Symbol.Flatten()()(Map("data" -> conv2))
    
    var fc = Symbol.FullyConnected(name = "fc")()(Map("data" -> conv2, "num_hidden" -> 1024))
    fc = Symbol.Activation()()(Map("data" -> fc, "act_type" -> "relu"))
    fc = Symbol.Dropout()()(Map("data" -> fc, "p" -> 0.5f))
    
    val out = Symbol.FullyConnected(name = "out")()(Map("data" -> fc, "num_hidden" -> nClass))
    Symbol.SoftmaxOutput("softmax")()(Map("data" -> out, "label" -> label))
  }

  // Create model
  def convNetSNN(batchSize: Int, dropoutProb: Float, nClass: Int): Symbol = {
    val data = Symbol.Variable("data")
    val label = Symbol.Variable("label")
    
    var conv1 = Symbol.Convolution("conv1")()(
        Map("data" -> data, "kernel" -> "(5, 5)", "pad" -> "(2, 2)", "num_filter" -> 32))
    conv1 = Symbol.Pooling("pool1")()(
        Map("data" -> conv1, "pool_type" -> "max", "kernel" -> "(2, 2)", "stride" -> "(2, 2)"))
  
    var conv2 = Symbol.Convolution("conv2")()(
        Map("data" -> conv1, "kernel" -> "(5, 5)", "pad" -> "(2, 2)", "num_filter" -> 64))
    conv2 = Symbol.Pooling("pool2")()(
        Map("data" -> conv2, "pool_type" -> "max", "kernel" -> "(2, 2)", "stride" -> "(2, 2)"))
    
    conv2 = Symbol.Flatten()()(Map("data" -> conv2))
    
    var fc = Symbol.FullyConnected(name = "fc")()(Map("data" -> conv2, "num_hidden" -> 1024))
    fc = Ops.selu(fc)
    fc = Ops.dropoutSelu(fc, rate = dropoutProb, xShape = Shape(batchSize, 1024))
    
    val out = Symbol.FullyConnected(name = "out")()(Map("data" -> fc, "num_hidden" -> nClass))
    Symbol.SoftmaxOutput("softmax")()(Map("data" -> out, "label" -> label))
  }

  def main(args: Array[String]): Unit = {
    val snst = new SNNs_CNN_MNIST
    val parser: CmdLineParser = new CmdLineParser(snst)
    try {
      parser.parseArgument(args.toList.asJava)
      assert(snst.dataPath != null)

      val ctx = if (snst.gpu >= 0) Context.gpu(snst.gpu) else Context.cpu()

      // Network Parameters
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

      val reluNet = convNetReLU(nClasses)
      val snnNetTrain = convNetSNN(snst.batchSize, 0.05f, nClasses)
      val snnNetTest = convNetSNN(snst.batchSize, 0f, nClasses)

      val execRelu = reluNet.simpleBind(ctx, gradReq = "write", shapeDict = Map("data" -> Shape(snst.batchSize, 1, 28, 28)))
      val execSNNTrain = snnNetTrain.simpleBind(ctx, gradReq = "write", shapeDict = Map("data" -> Shape(snst.batchSize, 1, 28, 28)))
      val execSNNTest = snnNetTest.simpleBind(ctx, gradReq = "write", shapeDict = Map("data" -> Shape(snst.batchSize, 1, 28, 28)))

      Random.normal(loc = 0f, scale = Math.sqrt(2.0 / 25).toFloat, out = execRelu.argDict("conv1_weight"))
      Random.normal(loc = 0f, scale = Math.sqrt(2.0 / (25 * 32)).toFloat, out = execRelu.argDict("conv2_weight")) 
      Random.normal(loc = 0f, scale = Math.sqrt(2.0 / (7 * 7 * 64)).toFloat, out = execRelu.argDict("fc_weight"))
      Random.normal(loc = 0f, scale = Math.sqrt(2.0 / 1024).toFloat, out = execRelu.argDict("out_weight"))
      Random.normal(loc = 0f, scale = 1e-6f, out = execRelu.argDict("conv1_bias"))
      Random.normal(loc = 0f, scale = 1e-6f, out = execRelu.argDict("conv2_bias"))
      Random.normal(loc = 0f, scale = 1e-6f, out = execRelu.argDict("fc_bias"))
      Random.normal(loc = 0f, scale = 1e-6f, out = execRelu.argDict("out_bias"))

      Random.normal(loc = 0f, scale = Math.sqrt(1.0 / 25).toFloat, out = execSNNTrain.argDict("conv1_weight"))
      Random.normal(loc = 0f, scale = Math.sqrt(1.0 / (25 * 32)).toFloat, out = execSNNTrain.argDict("conv2_weight")) 
      Random.normal(loc = 0f, scale = Math.sqrt(1.0 / (7 * 7 * 64)).toFloat, out = execSNNTrain.argDict("fc_weight"))
      Random.normal(loc = 0f, scale = Math.sqrt(1.0 / 1024).toFloat, out = execSNNTrain.argDict("out_weight"))
      Random.normal(loc = 0f, scale = 1e-6f, out = execSNNTrain.argDict("conv1_bias"))
      Random.normal(loc = 0f, scale = 1e-6f, out = execSNNTrain.argDict("conv2_bias"))
      Random.normal(loc = 0f, scale = 1e-6f, out = execSNNTrain.argDict("fc_bias"))
      Random.normal(loc = 0f, scale = 1e-6f, out = execSNNTrain.argDict("out_bias"))

      val opt = new RMSProp(learningRate = snst.lr)
      val snnParamsGrads = execSNNTrain.gradDict.filter{ case (n, d) => n != "data" && n != "label" }
                                                          .toList.zipWithIndex.map { case ((name, grad), idx) =>
        (idx, name, grad, opt.createState(idx, execSNNTrain.argDict(name)))
      }          
     val opt2 = new RMSProp(learningRate = snst.lr)                              
     val reluParamsGrads = execRelu.gradDict.filter{ case (n, d) => n != "data" && n != "label" }
                                                          .toList.zipWithIndex.map { case ((name, grad), idx) =>
        (idx, name, grad, opt.createState(idx, execRelu.argDict(name)))
      }

      val evalMetricSNN = new Accuracy
      val evalMetricRelu = new Accuracy
      
      val snnTrainLosses = ArrayBuffer[Float]()
      val reluTrainLosses = ArrayBuffer[Float]()

      for (epoch <- 0 until snst.trainEpoch) {
        evalMetricSNN.reset()
        evalMetricRelu.reset()

        var nBatch = 0
        var epochDone = false

        trainIter.reset()
        while (!epochDone) {
          var doReset = true
          while (doReset && trainIter.hasNext) {
            val dataBatch = trainIter.next()
            execSNNTrain.argDict("data").set(dataBatch.data(0).reshape(Shape(snst.batchSize, 1, 28, 28)))
            execSNNTrain.argDict("label").set(dataBatch.label(0))

            execSNNTrain.forward(isTrain = true)
            execSNNTrain.backward()

            snnParamsGrads.foreach { case (idx, name, grad, optimState) =>
              opt.update(idx, execSNNTrain.argDict(name), grad, optimState)
            }
            evalMetricSNN.update(dataBatch.label, execSNNTrain.outputs.take(1))
            
            execRelu.argDict("data").set(dataBatch.data(0).reshape(Shape(snst.batchSize, 1, 28, 28)))
            execRelu.argDict("label").set(dataBatch.label(0))

            execRelu.forward(isTrain = true)
            execRelu.backward()

            reluParamsGrads.foreach { case (idx, name, grad, optimState) =>
              opt2.update(idx, execRelu.argDict(name), grad, optimState)
            }
            evalMetricRelu.update(dataBatch.label, execRelu.outputs.take(1))
            
            nBatch += 1
          }
          if (doReset) {
            trainIter.reset()
          }
          epochDone = true
        }
        val (name, value) = evalMetricSNN.get
        name.zip(value).foreach { case (n, v) =>
          println(s"Epoch[$epoch] SNN Train-accuracy=$v")
        }
        val (names, values) = evalMetricRelu.get
        names.zip(values).foreach { case (n, v) =>
          println(s"Epoch[$epoch] ReLU Train-accuracy=$v")
        }

      }
      
        execSNNTest.argDict.foreach { case (name, nd) =>
          if (execSNNTrain.argDict.contains(name)) nd.set(execSNNTrain.argDict(name))
        }
        evalMetricSNN.reset()
        evalMetricRelu.reset()
        testIter.reset()
        while (testIter.hasNext) {
          val evalBatch = testIter.next()
          
          execSNNTest.argDict("data").set(evalBatch.data(0).reshape(Shape(snst.batchSize, 1, 28, 28)))
          execSNNTest.argDict("label").set(evalBatch.label(0))
          execSNNTest.forward(isTrain = false)
          evalMetricSNN.update(evalBatch.label, execSNNTest.outputs.take(1))
          
          execRelu.argDict("data").set(evalBatch.data(0).reshape(Shape(snst.batchSize, 1, 28, 28)))
          execRelu.argDict("label").set(evalBatch.label(0))
          execRelu.forward(isTrain = false)
          evalMetricRelu.update(evalBatch.label, execRelu.outputs.take(1)) 
          
          evalBatch.dispose()
        }
        val (names, values) = evalMetricSNN.get
        names.zip(values).foreach { case (n, v) =>
          println(s"SNN Validation-accuracy=$v")
      }
      val (name, value) = evalMetricRelu.get
      name.zip(value).foreach { case (n, v) =>
          println(s"ReLU Validation-accuracy=$v")
      }
    
      execSNNTrain.dispose()
      execSNNTest.dispose()
      execRelu.dispose()
      
    } catch {
      case ex: Exception => {
        println(ex.getMessage, ex)
        parser.printUsage(System.err)
        sys.exit(1)
      }
    }
  }
}

class SNNs_CNN_MNIST {
  @Option(name = "--batch-size", usage = "the training batch size, default 100")
  private var batchSize: Int = 128
  @Option(name = "--data-path", usage = "the mnist data path")
  private var dataPath: String = null
  @Option(name = "--gpu", usage = "which gpu card to use, default is -1, means using cpu")
  private var gpu: Int = -1
  @Option(name = "--lr", usage = "init learning rate")
  private var lr: Float = 0.001f
  @Option(name = "--train-epoch", usage = "training epoch")
  private var trainEpoch: Int = 10
}
