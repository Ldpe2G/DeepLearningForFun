import scala.collection.JavaConverters._
import org.kohsuke.args4j.CmdLineParser
import ml.dmlc.mxnet.Context
import utils.Visualizer
import models.Architectures
import utils.DataProcess
import ml.dmlc.mxnet.NDArray
import org.opencv.highgui.Highgui

/**
 * @author Depeng Liang
 */
object TestCycleGan {
  nu.pattern.OpenCV.loadShared()
  
  def main(args: Array[String]): Unit = {
    val opt = new utils.Options.OptTest()
    val parser: CmdLineParser = new CmdLineParser(opt)
    try {
      parser.parseArgument(args.toList.asJava)
      require(opt.modelPath != null && opt.inputImage != null && opt.outputPath != null)

      val ctx = if (opt.gpu == -1) Context.cpu() else Context.gpu(opt.gpu)

      val img = DataProcess.preprocessSingleImage4Test(opt.inputImage, ctx)
      
      val netGA = Architectures.defineGResNet6Blocks(opt.outputNC, opt.ngf, opt.norm)
      val netDA = Architectures.defineDNLayers(opt.ndf, normType = opt.norm)
      val netGB = Architectures.defineGResNet6Blocks(opt.outputNC, opt.ngf, opt.norm)
      val netDB = Architectures.defineDNLayers(opt.ndf, normType = opt.norm)

      val netG = if (opt.whichDirection == "AtoB") netGA else netGB

      val executor = netG.simpleBind(ctx, shapeDict = Map("gData" -> img.shape))
      val pretrain = NDArray.load2Map(opt.modelPath)
      executor.argDict.filter(_._1 != "gData").foreach { case (name, nd) =>
        val key = s"arg:$name"
        if(pretrain.contains(key)) nd.set(pretrain(key))
      }
     executor.auxDict.foreach { case (name, nd) =>
        val key = s"aux:$name"
        if(pretrain.contains(key)) nd.set(pretrain(key))
      }

      executor.argDict("gData").set(img)
      executor.forward(isTrain = false)
      val result = Visualizer.imShow(executor.outputs.head, show = false)
      Highgui.imwrite(s"${opt.outputPath}/result.png", result)

      sys.exit()
    } catch {
      case ex: Exception => {
        println(ex.getMessage, ex)
        parser.printUsage(System.err)
        sys.exit(1)
      }
    }
  }

}