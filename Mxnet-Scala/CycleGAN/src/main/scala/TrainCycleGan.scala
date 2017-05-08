
import scala.collection.JavaConverters._
import org.kohsuke.args4j.CmdLineParser
import utils.ImageIter
import ml.dmlc.mxnet.Context
import utils.Visualizer
import models.CycleGANModel
import models.Architectures
import utils.DataProcess
import ml.dmlc.mxnet.NDArray
import org.opencv.highgui.Highgui

/**
 * @author Depeng Liang
 */
object TrainCycleGan {

  nu.pattern.OpenCV.loadShared()
  
  def visualizeCurrentResults(model: CycleGANModel, opt: utils.Options.OptTrain, epoch: Int, counter: Int): Unit = {
    val (realA, fakeB, recA, realB, fakeA, recB) = model.GetCurrentVisuals()
    val realAM = Visualizer.imShow(realA, "realA")
    val fakeBM = Visualizer.imShow(fakeB, "fakeB")
    val recAM = Visualizer.imShow(recA, "recA")
    val realBM = Visualizer.imShow(realB, "realB")
    val fakeAM = Visualizer.imShow(fakeA, "fakeA")
    val recBM = Visualizer.imShow(recB, "recB")
    Visualizer.saveImgs(Array(realAM, fakeBM, recAM, realBM, fakeAM, recBM), opt.checkpointsDir, epoch, counter)
  }

  def main(args: Array[String]): Unit = {
    val opt = new utils.Options.OptTrain()
    val parser: CmdLineParser = new CmdLineParser(opt)
    try {
      parser.parseArgument(args.toList.asJava)
      require(opt.domainBPath != null && opt.domainAPath != null && opt.checkpointsDir != null)
      require(opt.batchSize == 1, "only support batch size 1 for now")
      if (opt.loadCheckpointsDir != null) require(opt.loadCheckpointsEpoch != 0)

      val ctx = if (opt.gpu == -1) Context.cpu() else Context.gpu(opt.gpu)
      
      val model = new CycleGANModel(opt, ctx)

      val iterA = new ImageIter(opt.domainAPath, opt, ctx, "realA")
      val iterB = new ImageIter(opt.domainBPath, opt, ctx, "realB")

      var counter = 0

      for (epoch <- opt.loadCheckpointsEpoch until opt.niter + opt.niterDecay) {
        
        while (iterA.hasNext && iterB.hasNext) {
          val realA = iterA.next().data(0)
          val realB = iterB.next().data(0)
          
          model.forwardBackward(realA, realB)
          
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
        iterA.reset()
        iterB.reset()

        if (counter % opt.saveEpochFreq == 0) {
          println(s"saving the latest model (epoch $epoch, iters $counter)")
          model.saveModel(prefix = "latest", epoch)
        }
        // update learning rate
        if (epoch > opt.niter) {
          model.UpdateLearningRate()
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
