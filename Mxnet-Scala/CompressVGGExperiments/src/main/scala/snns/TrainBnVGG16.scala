package snns

import org.kohsuke.args4j.{CmdLineParser, Option}
import scala.collection.JavaConverters._
import org.apache.mxnet.Context

class TrainBnVGG16 {
  @Option(name = "--batch-size", usage = "the training batch size, default 100")
  private var batchSize: Int = 128
  @Option(name = "--data-path", usage = "the Cifar data path")
  private var dataPath: String = null
  @Option(name = "--gpu", usage = "which gpu card to use, default is -1, means using cpu")
  private var gpu: Int = -1
  @Option(name = "--lr", usage = "init learning rate")
  private var lr: Float = 0.001f
  @Option(name = "--train-epoch", usage = "training epoch")
  private var trainEpoch: Int = 10
}

/**
 * @author Depeng Liang
 */
object TrainBnVGG16 {

  def main(args: Array[String]): Unit = {
    val opt = new TrainBnVGG16
    val parser: CmdLineParser = new CmdLineParser(opt)
    try {
      parser.parseArgument(args.toList.asJava)
      assert(opt.dataPath != null)

      val ctx = if (opt.gpu >= 0) Context.gpu(opt.gpu) else Context.cpu()

      
      
    } catch {
      case ex: Exception => {
        println(ex.getMessage, ex)
        parser.printUsage(System.err)
        sys.exit(1)
      }
    }
  }
}

