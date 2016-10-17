
import org.kohsuke.args4j.{CmdLineParser, Option}
import scala.collection.JavaConverters._
import ml.dmlc.mxnet.NDArray
import ml.dmlc.mxnet.Shape
import ml.dmlc.mxnet.Context
import ml.dmlc.mxnet.DataBatch
import ml.dmlc.mxnet.Symbol
import ml.dmlc.mxnet.Executor
import java.io.File
import javax.imageio.ImageIO
import scala.util.Random
import ml.dmlc.mxnet.optimizer.Adam
import ml.dmlc.mxnet.FactorScheduler
import org.sameersingh.scalaplot.MemXYSeries
import org.sameersingh.scalaplot.XYData
import org.sameersingh.scalaplot.XYChart
import scala.collection.mutable.ArrayBuffer
import org.sameersingh.scalaplot.gnuplot.GnuplotPlotter

/**
 * @author Depeng Liang
 */
object Train {

  def getTvGradExecutor(img: NDArray, ctx: Context, tvWeight: Float): Executor = {
    // create TV gradient executor with input binded on img
    if (tvWeight <= 0.0f) null

    val nChannel = img.shape(1)
    val sImg = Symbol.Variable("img")
    val sKernel = Symbol.Variable("kernel")
    val channels = Symbol.SliceChannel()(sImg)(Map("num_outputs" -> nChannel))
    val out = Symbol.Concat()((0 until nChannel).map { i =>
      Symbol.Convolution()()(Map("data" -> channels.get(i), "weight" -> sKernel,
                    "num_filter" -> 1, "kernel" -> "(3, 3)", "pad" -> "(1, 1)",
                    "no_bias" -> true, "stride" -> "(1, 1)"))
    }.toArray: _*)() * tvWeight
    val kernel = {
      val tmp = NDArray.empty(Shape(1, 1, 3, 3), ctx)
      tmp.set(Array[Float](0, -1, 0, -1, 4, -1, 0, -1, 0))
      tmp / 0.8f
    }
    out.bind(ctx, Map("img" -> img, "kernel" -> kernel))
  }

  def main(args: Array[String]): Unit = {
    val stin = new Train()
    val parser: CmdLineParser = new CmdLineParser(stin)
    try {
      parser.parseArgument(args.toList.asJava)
      require(stin.dataPath != null
          && stin.vggModelPath != null
          && stin.saveModelPath != null
          && stin.styleImage != null)
      
     require(stin.batchSize == 1, "now only support batch size 1")
      
      // params
      val vggParams = NDArray.load2Map(stin.vggModelPath)
      val styleWeight = stin.styleWeight
      val contentWeight = stin.contentWeight
      val dShape = Shape(stin.batchSize, 3, stin.imageSize, stin.imageSize)
      val clipNorm = 0.05f * dShape.product
      val modelPrefix = "resdual"
      val ctx = if (stin.gpu == -1) Context.cpu() else Context.gpu(stin.gpu)

      // init style
      val styleNp = DataProcessing.preprocessStyleImage(stin.styleImage, dShape, ctx)
      var styleMod = Basic.getStyleModule("style", dShape, ctx, vggParams)

      styleMod.forward(Array(styleNp))
      val styleArray = styleMod.getOutputs().map(_.copyTo(Context.cpu()))
      styleMod.dispose()
      styleMod = null

      // content
      val contentMod = Basic.getContentModule("content", dShape, ctx, vggParams)

      // loss
      val (loss, gScale) = Basic.getLossModule("loss", dShape, ctx, vggParams)
      val extraArgs = (0 until styleArray.length)
                                  .map( i => s"target_gram_$i" -> styleArray(i)).toMap
      loss.setParams(extraArgs)
      var gradArray = Array[NDArray]()
      for (i <- 0 until styleArray.length) {
        gradArray = gradArray :+ (NDArray.ones(Shape(1), ctx) * (styleWeight / gScale(i)))
      }
      gradArray = gradArray :+ (NDArray.ones(Shape(1), ctx) * contentWeight)

      // generator
      val generator = ResdualModel.getModule("res", dShape, ctx)
      if (stin.resumeModelPath != null) {
        generator.loadParams(stin.resumeModelPath)
      }
      
      val optimizer = new Adam(
          learningRate = stin.lr,
          wd = 0.005f)
      generator.initOptimizer(optimizer)

      var filelist = Random.shuffle(new File(stin.dataPath).list().toList)
      val numImage = filelist.length
      println(s"Dataset size: $numImage")

      val tvWeight = stin.tvWeight

      val startEpoch = 0
      val endEpoch = 3

      val trainLosses = ArrayBuffer[Float]()

      val tmpData = NDArray.empty(ctx, dShape.toArray: _*)
      // train
      for (i <- startEpoch until endEpoch) {
        filelist = Random.shuffle(filelist)
        for (idx <- 0 until filelist.length by stin.batchSize) {
          var dataArray = Array[NDArray]()

          val datas = (idx until idx + stin.batchSize).map { i => 
            DataProcessing.preprocessContentImage(s"${stin.dataPath}/${filelist(i)}", dShape, ctx)
          }
          
          tmpData.set(datas.foldLeft(Array[Float]()) { (acc, elem) => acc ++ elem.toArray})
          
          
          dataArray = dataArray :+ tmpData
          // get content
          contentMod.forward(Array(tmpData))
          // set target content
          loss.setParams(Map("target_content" -> contentMod.getOutputs()(0)))
          // gen_forward
          generator.forward(dataArray.takeRight(1))
          dataArray = dataArray :+ generator.getOutputs()(0)
          // loss forward
          loss.forward(dataArray.takeRight(1))
          loss.backward(gradArray)
          val lossGrad = loss.getInputGrads()(0)

          val grad = NDArray.zeros(tmpData.shape, ctx)
          
          val tvGradExecutor = getTvGradExecutor(generator.getOutputs()(0), ctx, tvWeight)
          tvGradExecutor.forward()
          grad += lossGrad + tvGradExecutor.outputs(0)
          val gNorm = NDArray.norm(grad)
          if (gNorm.toScalar > clipNorm) {
            grad *= clipNorm / gNorm.toScalar
          }
          generator.backward(Array(grad))
          generator.update()
          gNorm.dispose()
          tvGradExecutor.dispose()
          
          grad.dispose()
          
          if (idx % 20 == 0) {
            println(s"Epoch $i: Image $idx")
            val n = NDArray.norm(loss.getInputGrads()(0))
            trainLosses += n.toScalar / dShape.product
            
            var xTrain = trainLosses.indices.map(_.toDouble * 20).toArray
            var yTrainL = trainLosses.toArray.map(_.toDouble)
      
            var series = new MemXYSeries(xTrain, yTrainL)
            var data = new XYData(series)
            var chart = new XYChart("Training grads over iterations", data)
            var plotter = new GnuplotPlotter(chart)
            plotter.pdf(s"${stin.drawLossPath}/", "grad")
            
            println(s"Data Norm : ${n.toScalar / dShape.product}")
            n.dispose()
          }
          if (idx % 1000 == 0) {
            generator.saveParams(
                  s"${stin.saveModelPath}/${modelPrefix}_" +
                  s"${"%04d".format(i)}-${"%07d".format(idx)}.params")
          }
          datas.foreach(_.dispose())
        }
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

class Train {
  @Option(name = "--data-path", usage = "the coco data path")
  private var dataPath: String = null
  @Option(name = "--vgg-model-path", usage = "the pretrained model to use: ['vgg']")
  private var vggModelPath: String = null
  @Option(name = "--resume-model-path", usage = "the resume model")
  private var resumeModelPath: String = null
  @Option(name = "--draw-loss-path", usage = "path to save the result of visualization of the training process")
  private var drawLossPath: String = "."
  @Option(name = "--save-model-path", usage = "the save model path")
  private var saveModelPath: String = null
  @Option(name = "--style-image", usage = "the style image")
  private var styleImage: String = null
  @Option(name = "--content-weight", usage = "the weight for the content image")
  private var contentWeight: Float = 15f
  @Option(name = "--style-weight", usage = "the weight for the style image")
  private var styleWeight: Float = 2f
  @Option(name = "--tv-weight", usage = "the magtitute on TV loss")
  private var tvWeight: Float = 0.01f
  @Option(name = "--max-num-epochs", usage = "the maximal number of training epochs")
  private var maxNumEpochs: Int = 4000
  @Option(name = "--image-size", usage = "teh size of images")
  private var imageSize: Int = 256
  @Option(name = "--lr", usage = "the initial learning rate")
  private var lr: Float = 0.0001f
  @Option(name = "--gpu", usage = "which gpu card to use 0,1,2, default is -1, means using cpu")
  private var gpu: Int = -1
  @Option(name = "--batchsize", usage = "training batch size")
  private var batchSize: Int = 1
}
