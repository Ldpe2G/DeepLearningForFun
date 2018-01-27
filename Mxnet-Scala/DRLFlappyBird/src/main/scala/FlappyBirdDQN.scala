
import org.kohsuke.args4j.{CmdLineParser, Option}
import scala.collection.JavaConverters._
import scala.util.Random
import org.opencv.imgproc.Imgproc
import org.opencv.core.Mat
import org.opencv.core.CvType
import org.opencv.core.Size
import org.opencv.highgui.Highgui
import ml.dmlc.mxnet.Context

/**
 * @author Depeng Liang
 */
object FlappyBirdDQN {
  nu.pattern.OpenCV.loadShared()
  
  // preprocess raw image to 80*80 gray image
  def preprocess(observation: Mat, height: Int, width: Int): Mat = {
    val tmp = new Mat()
    Imgproc.resize(observation, tmp, new Size(width, height))
    val tmp2 = new Mat()
    Imgproc.cvtColor(tmp, tmp2, Imgproc.COLOR_RGB2GRAY)
    Imgproc.threshold(tmp2, tmp2, 1, 255, Imgproc.THRESH_BINARY)
    val result = new Mat(tmp2.rows(), tmp2.cols(), CvType.CV_32F)
    tmp2.convertTo(result, CvType.CV_32F, 1f)
    result
  }
  
  var root: String = null
  
  def main(args: Array[String]): Unit = {
    val pyrd = new FlappyBirdDQN()
    val parser: CmdLineParser = new CmdLineParser(pyrd)
    try {
      parser.parseArgument(args.toList.asJava)
      
      assert(pyrd.resourcesPath != null)
      root = pyrd.resourcesPath

      val ctx = if (pyrd.gpu == -1) Context.cpu() else Context.gpu(pyrd.gpu)
      
      val resizeH = 80
      val resizeW = 80
      
      val brain = new BrainDQNMx.BrainDQN(ctx, resizeH, resizeW, pyrd.saveModelPath, pyrd.resumeModelPath) 
      val (observation0, reward0, terminal0) = FlappyBird.frameStep(0)
      val observationBinary = preprocess(observation0, resizeH, resizeW)
      brain.setInitState(observationBinary)
	    
      while (true) {
        val action = brain.getAction()
        val (nextObservation, reward, terminal) = FlappyBird.frameStep(action)
        val nextObservationBinary = preprocess(nextObservation, resizeH, resizeW)
	brain.setPerception(nextObservationBinary, action, reward, terminal)
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

class FlappyBirdDQN {
  @Option(name = "--save-model-path", usage = "the output models path")
  private var saveModelPath: String = null
  @Option(name = "--resume-model-path", usage = "the resume model")
  private var resumeModelPath: String = null
  @Option(name = "--gpu", usage = "which gpu card to use, default is -1, means using cpu")
  private var gpu: Int = -1
  @Option(name = "--resources-path", usage = "the resource images path")
  private var resourcesPath: String = null
  
}

