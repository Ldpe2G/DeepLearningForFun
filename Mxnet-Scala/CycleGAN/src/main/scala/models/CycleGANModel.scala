package models

import ml.dmlc.mxnet.optimizer.Adam
import ml.dmlc.mxnet.Normal
import ml.dmlc.mxnet.Shape
import ml.dmlc.mxnet.NDArray
import ml.dmlc.mxnet.Context
import ml.dmlc.mxnet.Executor
import ml.dmlc.mxnet.Model
import ml.dmlc.mxnet.optimizer.RMSProp
import utils.DataProcess
import utils.Visualizer
import org.opencv.highgui.Highgui
import ml.dmlc.mxnet.Uniform

/**
 * @author Depeng Liang
 */
class CycleGANModel(opt: utils.Options.OptTrain, ctx: Context) {
  val realLabel = 1f
  val fakeLabel = 0f

  // def net and loss symbols
  val netGA = Architectures.defineGResNet6Blocks(opt.outputNC, opt.ngf, opt.norm)
  val netDA = Architectures.defineDNLayers(opt.ndf, normType = opt.norm)
  val netGB = Architectures.defineGResNet6Blocks(opt.outputNC, opt.ngf, opt.norm)
  val netDB = Architectures.defineDNLayers(opt.ndf, normType = opt.norm)

  val initializer = new Normal(sigma = 0.02f)

  val netDGInShape = Shape(opt.batchSize, opt.inputNC, opt.cropSize, opt.cropSize)
  // setup netGA
  val netGAExecutor = netGA.simpleBind(ctx, gradReq = "write", shapeDict = Map("gData" -> netDGInShape))
  require(netDGInShape == netGAExecutor.outputs.head.shape)
  // init params
  netGAExecutor.argDict.foreach { case (name, nd) => if (name != "gData") initializer(name, nd) }
  val tmpGradGA = netGAExecutor.gradDict.toArray.map { case (name, nd) => name -> nd.copy() }.toMap
  
  // setup netDA
  val netDAExecutor = netDA.simpleBind(ctx, gradReq = "write", shapeDict = Map("dData" -> netDGInShape))
  // init params
  netDAExecutor.argDict.foreach { case (name, nd) => if (name != "dData") initializer(name, nd) }
  val tmpGradDA = netDAExecutor.gradDict.toArray.map { case (name, nd) => name -> nd.copy() }.toMap

  // setup netGB
  val netGBExecutor = netGB.simpleBind(ctx, gradReq = "write", shapeDict = Map("gData" -> netDGInShape))
  require(netDGInShape == netGBExecutor.outputs.head.shape)
  // init params
  netGBExecutor.argDict.foreach { case (name, nd) => if (name != "gData") initializer(name, nd) }
  val tmpGradGB = netGBExecutor.gradDict.toArray.map { case (name, nd) => name -> nd.copy() }.toMap

  // setup netDB
  val netDBExecutor = netDB.simpleBind(ctx, gradReq = "write", shapeDict = Map("dData" -> netDGInShape))
  // init params
  netDBExecutor.argDict.foreach { case (name, nd) => if (name != "dData") initializer(name, nd) }
  val tmpGradDB = netDBExecutor.gradDict.toArray.map { case (name, nd) => name -> nd.copy() }.toMap
  require(netDBExecutor.outputs.head.shape == netDAExecutor.outputs.head.shape)
  
  if (opt.loadCheckpointsDir != null) {
    val dACheckpointName = "%s-%04d.params".format("latest-netDA", opt.loadCheckpointsEpoch)
    val dAPretrainedParams = NDArray.load2Map(s"${opt.loadCheckpointsDir}/$dACheckpointName")
    netDAExecutor.argDict.filter(_._1 != "dData").foreach { case (name, nd) =>
      val key = s"arg:$name"
      if(dAPretrainedParams.contains(key)) nd.set(dAPretrainedParams(key))
    }
    netDAExecutor.auxDict.foreach { case (name, nd) =>
      val key = s"aux:$name"
      if(dAPretrainedParams.contains(key)) nd.set(dAPretrainedParams(key))
    }
    dAPretrainedParams.foreach(_._2.dispose())
    
    val gACheckpointName = "%s-%04d.params".format("latest-netGA", opt.loadCheckpointsEpoch)
    val gAPretrainedParams = NDArray.load2Map(s"${opt.loadCheckpointsDir}/$gACheckpointName")
    netGAExecutor.argDict.filter(_._1 != "gData").foreach { case (name, nd) =>
      var key = s"arg:$name"
      if(gAPretrainedParams.contains(key)) nd.set(gAPretrainedParams(key))
    }
    netGAExecutor.auxDict.foreach { case (name, nd) =>
      val key = s"aux:$name"
      if(gAPretrainedParams.contains(key)) nd.set(gAPretrainedParams(key))
    }
    gAPretrainedParams.foreach(_._2.dispose())

    val dBCheckpointName = "%s-%04d.params".format("latest-netDB", opt.loadCheckpointsEpoch)
    val dBPretrainedParams = NDArray.load2Map(s"${opt.loadCheckpointsDir}/$dBCheckpointName")
    netDBExecutor.argDict.filter(_._1 != "dData").foreach { case (name, nd) =>
      var key = s"arg:$name"
      if(dBPretrainedParams.contains(key)) nd.set(dBPretrainedParams(key))
    }
    netDBExecutor.auxDict.foreach { case (name, nd) =>
      val key = s"aux:$name"
      if(dBPretrainedParams.contains(key)) nd.set(dBPretrainedParams(key))
    }
    dBPretrainedParams.foreach(_._2.dispose())

    val gBCheckpointName = "%s-%04d.params".format("latest-netGB", opt.loadCheckpointsEpoch)
    val gBPretrainedParams = NDArray.load2Map(s"${opt.loadCheckpointsDir}/$gBCheckpointName")
    netGBExecutor.argDict.filter(_._1 != "gData").foreach { case (name, nd) =>
      var key = s"arg:$name"
      if(gBPretrainedParams.contains(key)) nd.set(gBPretrainedParams(key))
    }
    netGBExecutor.auxDict.foreach { case (name, nd) =>
      val key = s"aux:$name"
      if(gBPretrainedParams.contains(key)) nd.set(gBPretrainedParams(key))
    }
    gBPretrainedParams.foreach(_._2.dispose())
  }

  val GANLoss = Architectures.getMSELoss(netDBExecutor.outputs.head.shape.product)
  val RecLoss = Architectures.getAbsLoss(netDGInShape.product)
  
  // setup GANLoss
  val GANLossExec = GANLoss.simpleBind(ctx, "write", shapeDict = Map("datas" -> netDAExecutor.outputs.head.shape))

  // setup RecLoss
  val RecLossExec = RecLoss.simpleBind(ctx, "write", shapeDict = Map("origin" -> netGAExecutor.outputs.head.shape))

  val optimizer =new Adam(learningRate = opt.lr, beta1 = opt.beta1)

  // init optimizer states
  var acc = 0
  val netGAStates = netGA.listArguments().zipWithIndex.map { case (name, idx) =>
    val state = optimizer.createState(idx, netGAExecutor.argDict(name))
    (idx, netGAExecutor.argDict(name), tmpGradGA(name), state)
  }
  acc += netGA.listArguments().length

  val netDAStates = netDA.listArguments().zipWithIndex.map { case (name, idx) =>
  val state = optimizer.createState(idx + acc, netDAExecutor.argDict(name))
    (idx + acc, netDAExecutor.argDict(name), tmpGradDA(name), state)
  }
  acc += netDA.listArguments().length

  val netGBStates = netGB.listArguments().zipWithIndex.map { case (name, idx) =>
  val state = optimizer.createState(idx + acc, netGBExecutor.argDict(name))
    (idx + acc, netGBExecutor.argDict(name), tmpGradGB(name), state)
  }
  acc += netGB.listArguments().length

  val netDBStates = netDB.listArguments().zipWithIndex.map { case (name, idx) =>
    val state = optimizer.createState(idx + acc, netDBExecutor.argDict(name))
    (idx + acc, netDBExecutor.argDict(name), tmpGradDB(name), state)
  }

  val realB = NDArray.empty(netDGInShape, ctx)
  val realA = NDArray.empty(netDGInShape, ctx)
  val fakeB = NDArray.empty(netGAExecutor.outputs.head.shape, ctx)
  val fakeA = NDArray.empty(netGBExecutor.outputs.head.shape, ctx)
  val recA = NDArray.empty(netGBExecutor.outputs.head.shape, ctx)
  val recB = NDArray.empty(netGAExecutor.outputs.head.shape, ctx)

  var errGA, errDA, errRecA, errGB, errDB, errRecB  = 0f
  val fakeAPool = new utils.ImagePool(opt.poolSize)
  val fakeBPool = new utils.ImagePool(opt.poolSize)


  def fGxBasic(netG: Executor, tmpGradG: Map[String, NDArray], netD: Executor, netE: Executor,
      real: NDArray, real2: NDArray, retFake: NDArray, retRec: NDArray,
      lambda1: Float, lambda2: Float): (Float, Float) = {
    
    netG.argDict("gData").set(real2)
    netG.forward(isTrain = true)
    this.RecLossExec.argDict("origin").set(real2)
    this.RecLossExec.argDict("rec").set(netG.outputs.head)
    this.RecLossExec.forward(isTrain = true)
    val errI = this.RecLossExec.outputs.head.toScalar * lambda2 * opt.identity
    this.RecLossExec.backward()
    val tmpI = this.RecLossExec.gradDict("rec") * lambda2 * opt.identity
    netG.backward(tmpI)
    netG.gradDict.foreach { case (name, grad) =>
      tmpGradG(name).set(grad)  
    }
    tmpI.dispose()

    // GAN loss: D_A(G_A(A))
    netG.argDict("gData").set(real)
    netG.forward(isTrain = true)
    retFake.set(netG.outputs.head)
    netD.argDict("dData").set(netG.outputs.head)
    netD.forward(isTrain = true)
    this.GANLossExec.argDict("datas").set(netD.outputs.head)
    this.GANLossExec.argDict("labels").set(realLabel)
    this.GANLossExec.forward(isTrain = true)
    val errG = this.GANLossExec.outputs.head.toScalar
    this.GANLossExec.backward()
    val dfDo1 = this.GANLossExec.gradDict("datas")
    netD.backward(dfDo1)
    val dfDGAN = netD.gradDict("dData")

    // forward cycle loss
    netE.argDict("gData").set(netG.outputs.head)
    netE.forward(isTrain = true)
    retRec.set(netE.outputs.head)
    this.RecLossExec.argDict("origin").set(real)
    this.RecLossExec.argDict("rec").set(netE.outputs.head)
    this.RecLossExec.forward(isTrain = true)
    val errRec = this.RecLossExec.outputs.head.toScalar * lambda1
    this.RecLossExec.backward()
    val tmp = this.RecLossExec.gradDict("rec") * lambda1
    netE.backward(tmp)
    val dfDoRec = netE.argDict("gData")
    tmp.dispose()

    val combineGrad = dfDGAN + dfDoRec
    netG.backward(combineGrad)
    netG.gradDict.foreach { case (name, grad) =>
      val sum = grad + tmpGradG(name)
      tmpGradG(name).set(sum)
      sum.dispose()
    }

    // back cycle loss
    netE.argDict("gData").set(real2)    
    netE.forward(isTrain = true)
    netG.argDict("gData").set(netE.outputs.head)
    netG.forward(isTrain = true)
    this.RecLossExec.argDict("origin").set(real2)
    this.RecLossExec.argDict("rec").set(netG.outputs.head)
    this.RecLossExec.forward(isTrain = true)
    this.RecLossExec.backward()
    val tmp2 = this.RecLossExec.gradDict("rec") * lambda2
    netG.backward(tmp2)
    netG.gradDict.foreach { case (name, grad) =>
      val sum = grad + tmpGradG(name)
      tmpGradG(name).set(sum)
      sum.dispose()
    }
    combineGrad.dispose()
    tmp2.dispose()
    (errG, errRec)
  }

  def fDxBasic(netD: Executor, tmpGradD: Map[String, NDArray], real: NDArray, fake: NDArray): Float = {
//    netD.gradDict.foreach(_._2.set(0f))

    // Real  log(D_A(B))
    netD.argDict("dData").set(real)
    netD.forward(isTrain = true)
    this.GANLossExec.argDict("datas").set(netD.outputs.head)
    this.GANLossExec.argDict("labels").set(realLabel)
    this.GANLossExec.forward(isTrain = true)
    val errDReal = this.GANLossExec.outputs.head.toScalar
    this.GANLossExec.backward()
    val dfDo = this.GANLossExec.gradDict("datas")
    netD.backward(dfDo)
    netD.gradDict.foreach { case (name, grad) =>
      tmpGradD(name).set(grad)  
    }
    
    // Fake log(1 - D_A(G_A(A)))
    netD.argDict("dData").set(fake)
    netD.forward(isTrain = true)
    this.GANLossExec.argDict("datas").set(netD.outputs.head)
    this.GANLossExec.argDict("labels").set(fakeLabel)
    this.GANLossExec.forward(isTrain = true)
    val errDFake = this.GANLossExec.outputs.head.toScalar
    this.GANLossExec.backward()
    val dfDo2 = this.GANLossExec.gradDict("datas")
    netD.backward(dfDo2)
    netD.gradDict.foreach { case (name, grad) =>
      val sum = grad + tmpGradD(name)
      tmpGradD(name).set(sum)
      sum.dispose()
    }
    // Compute loss
    val errD = (errDReal + errDFake) / 2
    errD
  }

  def forwardBackward(realANd: NDArray, realBNd: NDArray): Unit = {
    this.realA.set(realANd)
    this.realB.set(realBNd)

    val (errGA, errRecA) = fGxBasic(
        netGAExecutor, tmpGradGA, netDAExecutor, netGBExecutor, realA, realB, fakeB, recA, opt.lambdaA, opt.lambdaB)
    this.errGA = errGA
    this.errRecA = errRecA

    // fDx
    val fakeBB = this.fakeBPool.query(fakeB)
    val errDA = fDxBasic(netDAExecutor, tmpGradDA, realB, fakeBB)
    this.errDA = errDA
    if (fakeBB != fakeB) fakeBB.dispose()

    val (errGB, errRecB) = fGxBasic(
        netGBExecutor, tmpGradGB, netDBExecutor, netGAExecutor, realB, realA, fakeA, recB, opt.lambdaB, opt.lambdaA)
    this.errGB = errGB
    this.errRecB = errRecB

    val fakeAA = this.fakeAPool.query(fakeA)
    val errDB = fDxBasic(netDBExecutor, tmpGradDB, realA, fakeAA)
    this.errDB = errDB
    if (fakeAA != fakeA) fakeAA.dispose()
    
    netGAStates.foreach { case (idx, weight, grad, state) => optimizer.update(idx, weight, grad, state) }
    netDAStates.foreach { case (idx, weight, grad, state) => optimizer.update(idx, weight, grad, state) }
    netGBStates.foreach { case (idx, weight, grad, state) => optimizer.update(idx, weight, grad, state) }
    netDBStates.foreach { case (idx, weight, grad, state) => optimizer.update(idx, weight, grad, state) }
  }

  def GetCurrentErrorDescription(): String = {
    "[A] G: %.4f  D: %.4f  Rec: %.4f || [B] G: %.4f D: %.4f Rec: %.4f".format(errGA, errDA, errRecA, errGB, errDB, errRecB)
  }

   def saveModel(prefix: String, epoch: Int): Unit = {
     Model.saveCheckpoint(s"${opt.checkpointsDir}/${prefix}-netDA", epoch,
       netDA, netDAExecutor.argDict, netDAExecutor.auxDict)
     Model.saveCheckpoint(s"${opt.checkpointsDir}/${prefix}-netGA", epoch,
       netGA, netGAExecutor.argDict, netGAExecutor.auxDict)
     Model.saveCheckpoint(s"${opt.checkpointsDir}/${prefix}-netDB", epoch,
       netDB, netDBExecutor.argDict, netDBExecutor.auxDict)
      Model.saveCheckpoint(s"${opt.checkpointsDir}/${prefix}-netGB", epoch,
       netGB, netGBExecutor.argDict, netGBExecutor.auxDict)
   }

   def GetCurrentVisuals(): (NDArray, NDArray, NDArray, NDArray, NDArray, NDArray) = {
     (realA, fakeB, recA, realB, fakeA, recB)
   }
   var factor = 1f
   def UpdateLearningRate(): Unit = {
     factor *= 0.999f
     optimizer.setLrScale((netDAStates ++ netGAStates ++ netDBStates ++ netGBStates).map(_._1 -> factor).toMap)
   }
}
