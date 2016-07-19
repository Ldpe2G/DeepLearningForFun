import ml.dmlc.mxnet.NDArray
import ml.dmlc.mxnet.Shape
import ml.dmlc.mxnet.Context
import scala.io.Source
import ml.dmlc.mxnet.Symbol
import ml.dmlc.mxnet.Xavier
import ml.dmlc.mxnet.optimizer.Adam
import java.io.PrintWriter
import ml.dmlc.mxnet.optimizer.RMSProp
import ml.dmlc.mxnet.optimizer.SGD

object SVDF {
  
  def getNet(maxUser: Int, maxItem: Int): Symbol = {
    val hidden = 300
    var user = Symbol.Variable("user")
    var item = Symbol.Variable("item")
    val score = Symbol.Variable("score")

    user = Symbol.Embedding()(Map("data" -> user, "input_dim" -> maxUser, "output_dim" -> 500))
    user = Symbol.Flatten()(Map("data" -> user))
    user = Symbol.FullyConnected()(Map("data" -> user, "num_hidden" -> hidden))
    item = Symbol.Embedding()(Map("data" -> item, "input_dim" -> maxItem, "output_dim" -> 500))
    item = Symbol.FullyConnected()(Map("data" -> item, "num_hidden" -> hidden))
    item = Symbol.Flatten()(Map("data" -> item))
    var pred = user * item
    pred = Symbol.sumAxis()(Map("data" -> pred, "axis" -> 1))
    pred = Symbol.Flatten()(Map("data" -> pred))
    pred = Symbol.LinearRegressionOutput()(Map("data" -> pred, "label" -> score))
    pred
  }
  
  def maxId(fname: String): (Int, Int) = {
    var mu = 0
    var mi = 0
    val lines = Source.fromFile(fname).mkString.split("\n")
    for (line <- lines) {
      val tks = line.trim().split("\t")
      if (tks.length == 4) {
        mu = Math.max(mu, tks(0).toInt)
        mi = Math.max(mi, tks(1).toInt)
      }
    }
    (mu + 1, mi + 1)
  }
  
  def getAllDatas(path: String, batchSize: Int): (Array[Array[Float]], Array[Array[Float]], Array[Array[Float]]) = {
    val lines = Source.fromFile(path).mkString.split("\n")
    val length = {
      val tmp = lines.length / batchSize
      tmp * batchSize
    }
    val grouped = lines.take(length).grouped(batchSize).toArray
    
    val allDatas = ((Array[Array[Float]](), Array[Array[Float]](), Array[Array[Float]]()) /: grouped){ (acc, elem) =>
      val (batU, batI, batS) = ((Array[Float](), Array[Float](), Array[Float]()) /: elem) { (a, e) => 
        val tks = e.trim().split("\t")
        (a._1 :+ tks(0).toFloat, a._2 :+ tks(1).toFloat, a._3 :+ tks(2).toFloat)
      }
      (acc._1 :+ batU, acc._2 :+ batI, acc._3 :+ batS)
    }
    allDatas
  }
  
  def RMSE(label: NDArray, pred: NDArray): Float = {
    val labelA = label.toArray
    val predA = pred.toArray
    labelA.zip(predA).map(x => x._1 - x._2).map(x => x * x).sum
  }
  
  def main(args: Array[String]): Unit = {
    val root = "./datas/ml-100k"
    val (maxUser, maxItem) = maxId(s"$root/u.data")
        
    val SVDNet = getNet(maxUser, maxItem)
    
    val ctx = Context.gpu(0)

    val batchSize = 100
    val (argShapes, outputShapes, auxShapes) = SVDNet.inferShape(
      Map("user" -> Shape(batchSize, 1), "item" -> Shape(batchSize, 1),
              "score" -> Shape(batchSize, 1)))
              
    val initializer = new Xavier(factorType = "in", magnitude = 2.34f)

    val argNames = SVDNet.listArguments()
    val argDict = argNames.zip(argShapes.map(NDArray.empty(_, ctx))).toMap
    val gradDict = argNames.zip(argShapes).filter { case (name, shape) =>
      name != "user" && name != "item" && name != "score"
    }.map(x => x._1 -> NDArray.empty(x._2, ctx) ).toMap
    
    argDict.foreach { case (name, ndArray) =>
      if (name != "user" && name != "item" && name != "score") {
        initializer.initWeight(name, ndArray)
      }
     }
    
        
    val user = argDict("user")
    val item = argDict("item")
    val score = argDict("score")
   
    val executor = SVDNet.bind(ctx, argDict, gradDict)
    
    val opt = new Adam(learningRate = 0.00005f, wd = 0.0001f)
    
    val paramsGrads = gradDict.toList.zipWithIndex.map { case ((name, grad), idx) =>
      (idx, name, grad, opt.createState(idx, argDict(name)))
    }
    
    val (users, items, scores) = getAllDatas(s"$root/u.data", batchSize)
    
    val trainNum = (users.length * 0.8).toInt
    val (trainUser, trainItem, trainScore) = (users.take(trainNum), items.take(trainNum), scores.take(trainNum))
    val (testUser, testItem, testScore) = (users.drop(trainNum), items.drop(trainNum), scores.drop(trainNum))

    var iter = 0
    var minTestRMES = 100f
    
    for (epoch <- 0 until 100000) {
      
      user.set(trainUser(iter))
      item.set(trainItem(iter))
      score.set(trainScore(iter))
      
      iter += 1
      if (iter >= trainUser.length) iter = 0
      
      executor.forward(isTrain = true)
      executor.backward()

      paramsGrads.foreach { case (idx, name, grad, optimState) =>
        opt.update(idx, argDict(name), grad, optimState)
      }
      
      println(s"iter $epoch, training RMSE: ${Math.sqrt(RMSE(score, executor.outputs(0)) / batchSize)}, minTestRMES: $minTestRMES")      
      
      if (epoch != 0 && epoch % 50 == 0) {
        
        val tmp = for (i <- 0 until testUser.length) yield {
          user.set(testUser(i))
          item.set(testItem(i))
          score.set(testScore(i))
          
          executor.forward(isTrain = false)
          RMSE(score, executor.outputs(0))
        }
        val testRMSE = Math.sqrt(tmp.toArray.sum / (testUser.length * batchSize))
        if (testRMSE < minTestRMES) minTestRMES = testRMSE.toFloat
      }
      
    }

  }
  
}