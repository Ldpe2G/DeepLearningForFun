
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
          && stce.inputImagesPath != null
          && stce.outputPath != null
          && stce.resizeH > 0
          && stce.resizeW > 0)
        
      val dShape = Shape(1, 3, stce.resizeH, stce.resizeW)
      val ctx = if (stce.gpu == -1) Context.cpu() else Context.gpu(stce.gpu)     
      
      // generator
      val gen = ResdualModel.getModule("res", dShape, ctx, true)
      gen.loadParams(s"${stce.modelPath}")
      
      val imgs = new File(stce.inputImagesPath).list()
      for (imgName <- imgs) {
        val img = Image(new File(s"${stce.inputImagesPath}/${imgName}"))
  
        var start = System.currentTimeMillis()
        var contentNp =
          DataProcessing.preprocessContentImage(s"${stce.inputImagesPath}/${imgName}", dShape, ctx)
        gen.forward(Array(contentNp))
        var end = System.currentTimeMillis()
        println(s"process time usage: ${(end - start).toFloat / 1000}s")
              
        val newImg = gen.getOutputs()(0)
        val result = {
          val tmp = DataProcessing.postprocessImage(newImg)
          if (tmp.width != img.width || tmp.height != img.height) {
            tmp.scaleTo(img.width, img.height)
          } else tmp
        }
        
        DataProcessing.saveImage(DataProcessing.postprocessImage(newImg),
            s"${stce.outputPath}/${imgName}", stce.guassianRadius)
        
        contentNp.dispose()
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

class FastNeuralStyle {
  @Option(name = "--model-path", usage = "the pretrain model")
  private var modelPath: String = null
  @Option(name = "--input-images-path", usage = "path to input images")
  private var inputImagesPath: String = null
  @Option(name = "--resizeh", usage = "")
  private var resizeH: Int = 256
  @Option(name = "--resizew", usage = "")
  private var resizeW: Int = 256
  @Option(name = "--output-path", usage = "the output result path")
  private var outputPath: String = null
  @Option(name = "--gpu", usage = "which gpu card to use, default is -1, means using cpu")
  private var gpu: Int = -1
  @Option(name = "--guassian-radius", usage = "the gaussian blur filter radius")
  private var guassianRadius: Int = 1
}
