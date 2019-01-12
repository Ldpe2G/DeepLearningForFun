import org.kohsuke.args4j.{CmdLineParser, Option}
import scala.collection.JavaConverters._
import org.apache.mxnet._
import org.apache.mxnet.module.Module
import org.apache.mxnet.optimizer.Adam
import org.slf4j.LoggerFactory

/**
 * @author Depeng Liang
 */
object TrainVGG {

  private val logger = LoggerFactory.getLogger(classOf[TrainVGG])
  
  def main(args: Array[String]): Unit = {
    val trgg = new TrainVGG
    val parser: CmdLineParser = new CmdLineParser(trgg)
    try {
      parser.parseArgument(args.toList.asJava)
      assert(trgg.dataPath != null)
      
      var sym = VggSym.getVgg16Symbol(numClasses = 10)
      var argParams: Map[String, NDArray] = null
      var auxParams: Map[String, NDArray] = null

      if (trgg.finetuneModuleEpoch > -1) {
        val (symm, argParamss, auxParamss) = Model.loadCheckpoint(trgg.finetuneModulePrefix, trgg.finetuneModuleEpoch)
        sym = symm
        argParams = argParamss
        auxParams = auxParamss
      }

      val dataShape = Shape(trgg.batchSize, 3, 32, 32)
      val context = if (trgg.gpu == -1) Context.cpu() else Context.gpu(trgg.gpu)

      val params = Map(
        "path_imgrec" -> s"${trgg.dataPath}/train.rec",
        "shuffle" -> "True",
        "data_shape" -> "(3, 32, 32)",
        "fill_value" -> "0",
        "batch_size" -> s"${trgg.batchSize}",
        "data_name" -> "data",
        "label_name" -> "softmax_label"
      )
      val trainData = IO.ImageRecordIter(params)
      
      val valParams = Map(
        "path_imgrec" -> s"${trgg.dataPath}/test.rec",
        "shuffle" -> "False",
        "data_shape" -> "(3, 32, 32)",
        "batch_size" -> s"${trgg.batchSize}",
        "data_name" -> "data",
        "label_name" -> "softmax_label"
      )
      val valData = IO.ImageRecordIter(valParams)
      
      val module = new Module(sym, dataNames = IndexedSeq("data"), labelNames = IndexedSeq("softmax_label"), contexts = context)
      import org.apache.mxnet.DataDesc._
      module.bind(trainData.provideDataDesc, Some(trainData.provideLabelDesc), forTraining = true)
      module.initParams(initializer = new Xavier(rndType = "uniform", factorType = "avg", magnitude = 2.34f),
                        argParams = argParams, auxParams = auxParams)
      module.initOptimizer(optimizer = new Adam(learningRate = trgg.lr, wd = 0.001f))

      val metric = new Accuracy()
      
      var bestAcc = 0f
      for (epoch <- trgg.finetuneModuleEpoch + 1 until trgg.trainEpoch) {
        metric.reset()
        val tic = System.currentTimeMillis

        var nBatch = 0
        while (trainData.hasNext) {
          val dataBatch = trainData.next()
          
          module.forwardBackward(dataBatch)
          module.update()
          module.updateMetric(metric, dataBatch.label)
          
          if (nBatch % 100 == 0) {
            val (name, value) = metric.get
            logger.info(s"Epoch[$epoch] Batch[$nBatch] Train-${name.head}=${value.head}")
          }
          
          dataBatch.dispose()
          
          nBatch += 1
          
        }

        // one epoch of training is finished
        val (name, value) = metric.get
        logger.info(s"Epoch[$epoch] Train-${name.head}=${value.head}")
        val toc = System.currentTimeMillis
        logger.info(s"Epoch[$epoch] Time cost=${toc - tic}")

        // sync aux params across devices
        val (argParamsSync, auxParamsSync) = module.getParams
        module.setParams(argParamsSync, auxParamsSync)

        // evaluation on validation set
        valData.reset()
        metric.reset()
        while (valData.hasNext) {
          val dataBatch = valData.next()
          
          module.forward(dataBatch, isTrain = Some(false))
          metric.update(dataBatch.label, module.getOutputsMerged())
          
          dataBatch.dispose()
        }
        val (name2, value2) = metric.get
        logger.info(s"Epoch[$epoch] Validation-${name2.head}=${value2.head}")
        if (value2.head > bestAcc) {
          bestAcc = value2.head
          module.saveCheckpoint(prefix = s"${trgg.modelPath}_acc_${bestAcc}", epoch)
        }
        // end of 1 epoch, reset the data-iter for another epoch
        trainData.reset()
      }
  
      
      
    } catch {
      case ex: Exception => {
        println(ex.getMessage, ex)
        parser.printUsage(System.err)
        sys.exit(1)
      }
    }
  }
}

class TrainVGG {
  @Option(name = "--batch-size", usage = "the training batch size, default 100")
  private var batchSize: Int = 128
  @Option(name = "--data-path", usage = "the cifar10 data path")
  private var dataPath: String = null
  @Option(name = "--finetune-model-prefix", usage = "the model save path")
  private var finetuneModulePrefix: String = null
  @Option(name = "--finetune-model-epoch", usage = "the model save path")
  private var finetuneModuleEpoch: Int = -1
  @Option(name = "--save-model-path", usage = "the model save path")
  private var modelPath: String = null
  @Option(name = "--gpu", usage = "which gpu card to use, default is -1, means using cpu")
  private var gpu: Int = -1
  @Option(name = "--lr", usage = "init learning rate")
  private var lr: Float = 0.01f
  @Option(name = "--train-epoch", usage = "training epoch")
  private var trainEpoch: Int = 10000
}
