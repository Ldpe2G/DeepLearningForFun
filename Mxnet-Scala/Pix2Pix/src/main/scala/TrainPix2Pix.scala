
import scala.collection.JavaConverters._
import org.kohsuke.args4j.CmdLineParser
import utils.ImageIter
import org.apache.mxnet.Context
import utils.Visualizer
import models.Pix2PixModel
import org.apache.mxnet.io.PrefetchingIter

/**
 * @author Depeng Liang
 */
object TrainPix2Pix {

  nu.pattern.OpenCV.loadShared()
  
  def visualizeCurrentResults(model: Pix2PixModel, opt: utils.Options.OptTrain, epoch: Int, counter: Int): Unit = {
    val (realA, fakeB, realB) = model.GetCurrentVisuals()
    val realAM = Visualizer.imShow(realA, "input")
    val fakeBM = Visualizer.imShow(fakeB, "output")
    val realBM = Visualizer.imShow(realB, "target")
    Visualizer.saveImgs(Array(realAM, fakeBM, realBM), opt.checkpointsDir, epoch, counter)
  }

  def main(args: Array[String]): Unit = {
    val opt = new utils.Options.OptTrain()
    val parser: CmdLineParser = new CmdLineParser(opt)
    try {
      parser.parseArgument(args.toList.asJava)
      require(opt.dataPath != null)
      require(opt.batchSize == 1, "only support batch size 1 for now")
      if (opt.loadCheckpointsDir != null) require(opt.loadCheckpointsEpoch != 0)

      val ctx = if (opt.gpu == -1) Context.cpu() else Context.gpu(opt.gpu)

      val model = new Pix2PixModel(opt, ctx)

      val iter = new ImageIter(opt.dataPath, opt, ctx, "realAB")
      val prefetchIter = new PrefetchingIter(IndexedSeq(iter))
      
      var counter = 0

      for (epoch <- opt.loadCheckpointsEpoch until opt.niter) {
        while (prefetchIter.hasNext) {
          val data = prefetchIter.next()
          val realA = data.data(0)
          val realB = data.data(1)

          model.createRealFake(realA, realB)
          model.forwardBackward()
          
          if (counter % opt.displayFreq == 0) {
            visualizeCurrentResults(model, opt, epoch, counter)
          }

          if (counter % opt.printFreq == 0) {
            println(s"Epoch: [$epoch] [$counter] \n${model.GetCurrentErrorDescription()}")
          }

          if (counter % opt.saveLatestFreq == 0 && counter > 0) {
            println(s"saving the latest model (epoch $epoch, iters $counter)")
            model.saveModel(prefix = "latest", epoch)
          }

          counter += 1
        }
        prefetchIter.reset()

        if (counter % opt.saveEpochFreq == 0) {
          println(s"saving the latest model (epoch $epoch, iters $counter)")
          model.saveModel(prefix = "latest", epoch)
        }
      }

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
