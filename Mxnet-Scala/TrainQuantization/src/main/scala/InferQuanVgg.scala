import org.kohsuke.args4j.{CmdLineParser, Option}
import scala.collection.JavaConverters._
import org.apache.mxnet._
import org.apache.mxnet.module.Module
import org.apache.mxnet.optimizer.Adam
import org.slf4j.LoggerFactory

/**
 * @author Depeng Liang
 */
object InferQuanVgg {

  private val logger = LoggerFactory.getLogger(classOf[InferQuanVgg])
  
  def main(args: Array[String]): Unit = {
    val trgg = new InferQuanVgg
    val parser: CmdLineParser = new CmdLineParser(trgg)
    try {
      parser.parseArgument(args.toList.asJava)
      assert(trgg.valDataPath != null && trgg.finetuneModuleEpoch > -1)

      val trainSym = VggQuanSym.getQuanVgg16Symbol(numClasses = 10)
      val sym = InferQuanVGGSym.getQuanVgg16Symbol(numClasses = 10, quantizeLevel = 128)

      val auxNames = trainSym.listAuxiliaryStates()
      val (argShapes, _, auxShapes) = trainSym.inferShape(Shape(trgg.batchSize, 3, 32, 32))
      val auxMaps = auxNames.zip(auxShapes).filter(e => !e._1.contains("moving_max_beta")).toMap
      
      val argNames = trainSym.listArguments()
      val argMaps = argNames.zip(argShapes)
                  .filter(e => !e._1.contains("moving_max_beta") && !e._1.contains("softmax_label")).toMap
      
      val saveDict = NDArray.load("%s-%04d.params".format(trgg.finetuneModulePrefix, trgg.finetuneModuleEpoch))
      val argParams = scala.collection.mutable.HashMap[String, NDArray]()
      val auxParams = scala.collection.mutable.HashMap[String, NDArray]()
      for ((k, v) <- saveDict._1 zip saveDict._2) {
        val splitted = k.split(":", 2)
        val tp = splitted(0)
        val name = splitted(1)
        if (tp == "arg") {
          argParams(name) = v
        } else if (tp == "aux") {
          auxParams(name) = v
        }
      }

      val dataShape = Shape(trgg.batchSize, 3, 32, 32)
      val context = if (trgg.gpu == -1) Context.cpu() else Context.gpu(trgg.gpu)

      val valParams = Map(
        "path_imgrec" -> s"${trgg.valDataPath}/test.rec",
        "shuffle" -> "False",
        "data_shape" -> "(3, 32, 32)",
        "batch_size" -> s"${trgg.batchSize}",
        "data_name" -> "data",
        "label_name" -> "softmax_label"
      )
      val valData = IO.ImageRecordIter(valParams)

      val module = sym.simpleBind(context, gradReq = "write", 
          shapeDict = Map("data" -> Shape(trgg.batchSize, 3, 32, 32)) ++: auxMaps ++: argMaps)
      argParams.foreach { case (k, v) => 
        if (module.argDict.contains(k)) v.copyTo(module.argDict(k))
      }
      auxParams.foreach { case (k, v) => 
        if (module.auxDict.contains(k)) v.copyTo(module.auxDict(k))
        if (module.argDict.contains(k)) v.copyTo(module.argDict(k))
      }
            
      val metric = new Accuracy()

      valData.reset()
      metric.reset()
      while (valData.hasNext) {
        val dataBatch = valData.next()
        
        dataBatch.data(0).copyTo(module.argDict("data"))
        module.forward(isTrain = false)
        metric.update(dataBatch.label, module.outputs)
        
        dataBatch.dispose()
      }
      val (name2, value2) = metric.get
      logger.info(s"Validation-${name2.head}=${value2.head}")
      
    } catch {
      case ex: Exception => {
        println(ex.getMessage, ex)
        parser.printUsage(System.err)
        sys.exit(1)
      }
    }
  }
}

class InferQuanVgg {
  @Option(name = "--batch-size", usage = "the training batch size, default 100")
  private var batchSize: Int = 128
  @Option(name = "--val-data-path", usage = "the cifar10 data path")
  private var valDataPath: String = null
  @Option(name = "--finetune-model-prefix", usage = "the model save path")
  private var finetuneModulePrefix: String = null
  @Option(name = "--finetune-model-epoch", usage = "the model save path")
  private var finetuneModuleEpoch: Int = -1
  @Option(name = "--gpu", usage = "which gpu card to use, default is -1, means using cpu")
  private var gpu: Int = -1
}
