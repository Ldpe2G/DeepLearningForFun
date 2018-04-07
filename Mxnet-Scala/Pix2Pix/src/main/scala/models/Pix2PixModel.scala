package models

import org.apache.mxnet.optimizer.Adam
import org.apache.mxnet.Shape
import org.apache.mxnet.NDArray
import org.apache.mxnet.Context
import org.apache.mxnet.Executor
import org.apache.mxnet.Model
import org.apache.mxnet.Random

/**
 * @author Depeng Liang
 */
class Pix2PixModel(opt: utils.Options.OptTrain, ctx: Context) {
  val realLabel = 1f
  val fakeLabel = 0f

  // def net and loss symbols
  val netG = Architectures.defineUnet(outputNC = opt.outputNC, opt.ngf)
  val netD = Architectures.defineDNLayers(ndf = opt.ndf, nLayers = 3)

  val netGInShape = Shape(opt.batchSize, opt.inputNC, opt.cropSize, opt.cropSize)
  // setup netGA
  val netGExecutor = netG.simpleBind(ctx, gradReq = "write", shapeDict = Map("gData" -> netGInShape))
  require(netGInShape == netGExecutor.outputs.head.shape)

  // init params
  netGExecutor.argDict.foreach { case (name, nd) => 
    if (name != "gData") {
      Random.normal(loc = 0f, scale = 0.02f, out = nd)
    }
  }
  val tmpGradG = netGExecutor.gradDict.toArray.map { case (name, nd) =>
    name -> nd.copy()
  }.toMap

  val netDInShape = Shape(opt.batchSize, opt.inputNC + opt.outputNC, opt.cropSize, opt.cropSize)
  // setup netDA
  val netDExecutor = netD.simpleBind(ctx, gradReq = "write", shapeDict = Map("dData" -> netDInShape))
  // init params
  netDExecutor.argDict.foreach { case (name, nd) =>
    if (name != "dData" && name != "dloss_label") {
      Random.normal(loc = 0f, scale = 0.02f, out = nd)
    }  
  }
  val tmpGradD = netDExecutor.gradDict.toArray.filter(_._1 != "dloss_label").map { case (name, nd) =>
    name -> nd.copy()
  }.toMap

  if (opt.loadCheckpointsDir != null) {
    val dCheckpointName = "%s-%04d.params".format("latest-netD", opt.loadCheckpointsEpoch)
    val dPretrainedParams = NDArray.load2Map(s"${opt.loadCheckpointsDir}/$dCheckpointName")
    netDExecutor.argDict.filter(_._1 != "dData").foreach { case (name, nd) =>
      val key = s"arg:$name"
      if(dPretrainedParams.contains(key)) nd.set(dPretrainedParams(key))
    }
    netDExecutor.auxDict.foreach { case (name, nd) =>
      val key = s"aux:$name"
      if(dPretrainedParams.contains(key)) nd.set(dPretrainedParams(key))
    }
    dPretrainedParams.foreach(_._2.dispose())
    
    val gCheckpointName = "%s-%04d.params".format("latest-netG", opt.loadCheckpointsEpoch)
    val gPretrainedParams = NDArray.load2Map(s"${opt.loadCheckpointsDir}/$gCheckpointName")
    netGExecutor.argDict.filter(_._1 != "gData").foreach { case (name, nd) =>
      var key = s"arg:$name"
      if(gPretrainedParams.contains(key)) nd.set(gPretrainedParams(key))
    }
    netGExecutor.auxDict.foreach { case (name, nd) =>
      val key = s"aux:$name"
      if(gPretrainedParams.contains(key)) nd.set(gPretrainedParams(key))
    }
    gPretrainedParams.foreach(_._2.dispose())
  }

  val RecLoss = Architectures.getAbsLoss()
  
  // setup RecLoss
  val RecLossExec = RecLoss.simpleBind(ctx, "write", shapeDict = Map("origin" -> netGExecutor.outputs.head.shape))

  val optimizer =new Adam(learningRate = opt.lr, beta1 = opt.beta1)

  // init optimizer states
  var acc = 0
  val netGStates = netG.listArguments().zipWithIndex.map { case (name, idx) =>
    val state = optimizer.createState(idx, netGExecutor.argDict(name))
    (idx, netGExecutor.argDict(name), tmpGradG(name), state)
  }
  acc += netG.listArguments().length
  val netDStates = netD.listArguments().filter(_ != "dloss_label").zipWithIndex.map { case (name, idx) =>
  val state = optimizer.createState(idx + acc, netDExecutor.argDict(name))
    (idx + acc, netDExecutor.argDict(name), tmpGradD(name), state)
  }

  val realB = NDArray.empty(netGInShape, ctx)
  val realA = NDArray.empty(netGInShape, ctx)
  val fakeB = NDArray.empty(netGExecutor.outputs.head.shape, ctx)
  val realAB = NDArray.empty(Shape(opt.batchSize, opt.inputNC + opt.outputNC, opt.cropSize, opt.cropSize), ctx)
  val fakeAB = NDArray.empty(Shape(opt.batchSize, opt.inputNC + opt.outputNC, opt.cropSize, opt.cropSize), ctx)

  var errG, errD, errL1  = 0f

  def createRealFake(realAND: NDArray, realBND: NDArray): Unit = {
    this.realA.set(realAND)
    this.realB.set(realBND)
    // create fake
    netGExecutor.argDict("gData").set(realA)
    netGExecutor.forward(isTrain = true)
    fakeB.set(netGExecutor.outputs.head)

    val tmp = NDArray.Concat(Map("dim" -> 1))(realA, realB)
    this.realAB.set(tmp)
    tmp.dispose()

    val tmp2 = NDArray.Concat(Map("dim" -> 1))(realA, fakeB)
    this.fakeAB.set(tmp2)
    tmp2.dispose()
  }

  
  def fDx(): Unit = {
    // Real
    netDExecutor.argDict("dData").set(this.realAB)
    netDExecutor.argDict("dloss_label").set(this.realLabel)
    netDExecutor.forward(isTrain = true)
    val errDReal = netDExecutor.outputs.head.toArray.sum
    netDExecutor.backward()
    netDExecutor.gradDict.filter(_._1 != "dloss_label").foreach { case (name, grad) =>
      tmpGradD(name).set(grad)  
    }

    // Fake
    netDExecutor.argDict("dData").set(fakeAB)
    netDExecutor.argDict("dloss_label").set(this.fakeLabel)
    netDExecutor.forward(isTrain = true)
    val errDFake = netDExecutor.outputs.head.toArray.sum
    netDExecutor.backward()
    netDExecutor.gradDict.filter(_._1 != "dloss_label").foreach { case (name, grad) =>
      val sum = grad + tmpGradD(name)
      tmpGradD(name).set(sum)
      sum.dispose()
    }

    // Compute loss
    errD = (errDReal + errDFake) / 2
  }

  def fGx(): Unit = {
    netDExecutor.argDict("dData").set(fakeAB)
    netDExecutor.argDict("dloss_label").set(this.realLabel)
    netDExecutor.forward(isTrain = true)
    errG = netDExecutor.outputs.head.toArray.sum
    netDExecutor.backward()
    val dfDg = NDArray.SliceChannel(Map("axis" -> 1, "num_outputs" -> 2))(netDExecutor.gradDict("dData"))(1)

    RecLossExec.argDict("origin").set(realB)
    RecLossExec.argDict("rec").set(fakeB)
    RecLossExec.forward(isTrain = true)
    errL1 = RecLossExec.outputs.head.toArray.sum
    RecLossExec.backward()
    val dfDoAE = RecLossExec.gradDict("rec") * opt.lambda
    
    val combine = dfDg + dfDoAE
    netGExecutor.backward(combine)
    netGExecutor.gradDict.foreach { case (name, grad) =>
      tmpGradG(name).set(grad)  
    }

    combine.dispose()
    dfDoAE.dispose()
    dfDg.dispose()
  }

  def forwardBackward(): Unit = {
    fDx()
    fGx()

    netDStates.foreach { case (idx, weight, grad, state) => optimizer.update(idx, weight, grad, state) }
    netGStates.foreach { case (idx, weight, grad, state) => optimizer.update(idx, weight, grad, state) }
  }

  def GetCurrentErrorDescription(): String = {
    "G: %.4f  D: %.4f  Rec: %.4f".format(errG, errD, errL1)
  }

def saveModel(prefix: String, epoch: Int): Unit = {
  Model.saveCheckpoint(s"${opt.checkpointsDir}/${prefix}-netD", epoch,
    netD, netDExecutor.argDict, netDExecutor.auxDict)
  Model.saveCheckpoint(s"${opt.checkpointsDir}/${prefix}-netG", epoch,
    netG, netGExecutor.argDict, netGExecutor.auxDict)
}

  def GetCurrentVisuals(): (NDArray, NDArray, NDArray) = {
    (realA, fakeB, realB)
  }
}
