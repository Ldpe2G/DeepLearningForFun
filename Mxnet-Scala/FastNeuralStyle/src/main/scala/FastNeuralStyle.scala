
import org.kohsuke.args4j.{CmdLineParser, Option}
import scala.collection.JavaConverters._
import org.apache.mxnet.Shape
import org.apache.mxnet.Context
import com.sksamuel.scrimage.Image
import java.io.File

/**
 * @author Depeng Liang
 */
object FastNeuralStyle {

  def main(args: Array[String]): Unit = {
    val stce = new FastNeuralStyle
    val parser: CmdLineParser = new CmdLineParser(stce)
    try {
      parser.parseArgument(args.toList.asJava)
      assert(stce.modelPath != null
          && stce.inputImage != null
          && stce.outputPath != null)

      val img = Image(new File(s"${stce.inputImage}"))

      val dShape = Shape(1, 3, img.height, img.width)
      val ctx = if (stce.gpu == -1) Context.cpu() else Context.gpu(stce.gpu)

      // generator
      val gen = ResdualModel.getModule("res", dShape, ctx)
      gen.loadParams(s"${stce.modelPath}")

      var start = System.currentTimeMillis()
      var contentNp =
        DataProcessing.preprocessContentImage(s"${stce.inputImage}", dShape, ctx)
      gen.forward(Array(contentNp))
      var end = System.currentTimeMillis()
      println(s"first time usage: ${(end - start).toFloat / 1000}s")

      start = System.currentTimeMillis()
      contentNp =
        DataProcessing.preprocessContentImage(s"${stce.inputImage}", dShape, ctx)
      gen.forward(Array(contentNp))
      end = System.currentTimeMillis()
      println(s"second time usage: ${(end - start).toFloat / 1000}s")
            
      val newImg = gen.getOutputs()(0)
      DataProcessing.saveImage(newImg, s"${stce.outputPath}/out.jpg", stce.guassianRadius)

    } catch {
      case ex: Exception => {
        println(ex.getMessage, ex)
        parser.printUsage(System.err)
        sys.exit(1)
      }
    }
  }
}

class FastNeuralStyle {
  @Option(name = "--model-path", usage = "the pretrain model")
  private var modelPath: String = null
  @Option(name = "--input-image", usage = "the style image")
  private var inputImage: String = null
  @Option(name = "--output-path", usage = "the output result path")
  private var outputPath: String = null
  @Option(name = "--gpu", usage = "which gpu card to use, default is -1, means using cpu")
  private var gpu: Int = -1
  @Option(name = "--guassian-radius", usage = "the gaussian blur filter radius")
  private var guassianRadius: Int = 1
}
