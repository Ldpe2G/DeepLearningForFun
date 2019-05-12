import org.apache.mxnet.Context
import org.opencv.core.Mat
import org.apache.mxnet.Symbol
import org.apache.mxnet.Shape
import org.apache.mxnet.Xavier
import org.apache.mxnet.optimizer.Adam
import org.apache.mxnet.DataBatch
import org.apache.mxnet.NDArray
import scala.util.Random
import scala.collection.mutable.ArrayBuffer
import java.io.File
import org.apache.mxnet.optimizer.RMSProp
import org.apache.mxnet.util.OptionConversion._

/**
 * @author Depeng Liang
 */
object BrainDQNMx {
  
  // Hyper Parameters:
  val ACTIONS = 2
  val GAMMA = 0.99f // decay rate of past observations
  val OBSERVE = 1000 // timesteps to observe before training
  val EXPLORE = 200000 // frames over which to anneal epsilon
  val REPLAY_MEMORY = 50000 // number of previous transitions to remember
  val BATCH_SIZE = 32 // size of minibatch
  val UPDATE_TIME = 100
  
  def dataPrep(data: Mat): Array[Float] = {
    val size = (data.total * data.channels).toInt
    val buff = new Array[Float](size)
    data.get(0, 0, buff)
    val re = buff.map(_ - 128)
    re
  }

  class BrainDQN(ctx: Context, screenHeight: Int, screenWidth: Int,
      saveModelPath: String = null, resumeModel: String = null) {
    // init replay memory
    private var replayMemory = Array[(Array[Float], Int, Float, Array[Float], Boolean)]()

    private var timeStep = 0
    private var currentState: Array[Float] = null
    private val target = createQNetwork(isTrain = false)

    private val Qnet = createQNetwork()
    if (resumeModel != null) {
      this.Qnet.loadParams(resumeModel)
    }
    copyTargetQNetwork()
    
    def sym(predict: Boolean = false): Symbol = {
      val data = Symbol.Variable("data")
      val yInput = Symbol.Variable("yInput")
      val actionInput = Symbol.Variable("actionInput")
      val conv1 = Symbol.api.Convolution(data, kernel = Shape(8,8), stride = Shape(4,4), pad = Shape(2,2), num_filter = 32, name = "conv1")
      val relu1 = Symbol.api.relu(conv1, name = "relu1")
      val pool1 = Symbol.api.Pooling(relu1, kernel = Shape(2,2), stride = Shape(2,2), pool_type = "max", name = "pool1")
      val conv2 = Symbol.api.Convolution(pool1, kernel = Shape(4,4), stride = Shape(2,2), pad = Shape(1,1), num_filter = 64, name = "conv2")
      val relu2 = Symbol.api.relu(conv2, name = "relu2")
      val conv3 = Symbol.api.Convolution(relu2, kernel = Shape(3,3), stride = Shape(1,1), pad = Shape(1,1), num_filter = 64, name = "conv3")
      val relu3 = Symbol.api.relu(conv3, name = "relu3")
      val flat  = Symbol.api.Flatten(relu3)
      val fc1 = Symbol.api.FullyConnected(flat, num_hidden = 512, name = "fc1")
      val relu4 = Symbol.api.relu(fc1, name = "relu4")
      val Qvalue = Symbol.api.FullyConnected(relu4, num_hidden = ACTIONS, name = "qvalue")
      val temp = Qvalue * actionInput
      val coeff = Symbol.api.sum(temp, axis = Shape(1), name = "temp1")
      val output = Symbol.pow(coeff - yInput, 2)
      val loss=Symbol.api.MakeLoss(output)
      
      if (predict) {
        Qvalue
      } else {
        loss
      }
    }
    
    def createQNetwork(isTrain: Boolean = true): Module = {
      val modQ = if (isTrain) {
        val module = new Module(
          symbol = sym(),
          context = ctx,
          dataShapes = Map("data" -> Shape(BATCH_SIZE, 4, screenHeight, screenWidth),
              "actionInput" -> Shape(BATCH_SIZE, ACTIONS)),
          labelShapes = Map("yInput" -> Shape(BATCH_SIZE)),
          initializer = new Xavier(factorType = "in", magnitude = 2.34f),
          forTraining = isTrain)
        
        module.initOptimizer(new Adam(
          learningRate = 0.0002f,
          wd = 0f,
          beta1 = 0.5f
        ))
        module
      } else {
        val module = new Module(
          symbol = sym(predict = true),
          context = ctx,
          dataShapes = Map("data" -> Shape(1, 4, screenHeight, screenWidth)),
          initializer = new Xavier(factorType = "in", magnitude = 2.34f),
          forTraining = isTrain)
        module
      }
      modQ
    }
    
    def copyTargetQNetwork(): Unit = {
      val (argParams, auxParams) = this.Qnet.getParams()
      this.target.setParams(argParams ++ auxParams)
    }

    def setInitState(observation: Mat): Unit = {
      val temp=dataPrep(observation)
      this.currentState = temp ++ temp ++ temp ++ temp
    }

    def getAction(): Int = {
      val ndArray = NDArray.empty(Shape(1, 4, screenHeight, screenWidth), ctx)
      ndArray.set(this.currentState)
      this.target.forward(Map("data" -> ndArray))
      val QValue = this.target.getOutputs()(0).copyTo(Context.cpu(0))
      val action = {
        val tmp = NDArray.argmax_channel(QValue)
        val result = tmp.toArray(0).toInt
        tmp.dispose()
        result
      }
      ndArray.dispose()
      QValue.dispose()
      action
    }
    
    def getRandomSamples: Array[(Array[Float], Int, Float, Array[Float], Boolean)] = {
      val tmp = replayMemory.toList
      val randomIdx = Random.shuffle(tmp.indices.toList).take(BATCH_SIZE)
      randomIdx.map(tmp(_)).toArray
    }
    
    val yBatch = NDArray.empty(Shape(BATCH_SIZE), ctx)
    val actionBatchArr = NDArray.empty(Shape(BATCH_SIZE, ACTIONS), ctx)
    val stateBatchArr = NDArray.empty(Shape(BATCH_SIZE, 4, screenHeight, screenWidth), ctx)
    val tmpData = NDArray.empty(Shape(1, 4, screenHeight, screenWidth), ctx)
    
    def trainQNetwork(): Unit = {
       // Step 1: obtain random minibatch from replay memory
      val miniBatch = getRandomSamples
      val stateBatch = miniBatch.map(_._1)
      val actionBatch = miniBatch.map(_._2).map { x =>
        val y = Array(0f, 0f)
        y(x) = 1f
        y
      }
      val rewardBatch = miniBatch.map(_._3)
      val nextStateBatch = miniBatch.map(_._4)
      
      // Step 2: calculate y 
      val Qvalue = ArrayBuffer[NDArray]()
      for (i <- 0 until BATCH_SIZE) {
        tmpData.set(nextStateBatch(i))
        this.target.forward(Map("data" -> tmpData))
        Qvalue.append(this.target.getOutputs()(0).copyTo(Context.cpu()))
      }
      val terminal = miniBatch.map(_._5)
      yBatch.set(rewardBatch)

      if (terminal.filter(_ == false).length > 0) {
        val idx = terminal.zipWithIndex.filter(_._1 == false).map(_._2)
        val yBatchArray = yBatch.toArray
        val maxArr = {
          val tmp = Qvalue.toArray.map(NDArray.max(_))
          val result = tmp.map(_.toArray(0))
          tmp.foreach(_.dispose())
          result
        }
        idx.foreach( i => yBatchArray(i) = yBatchArray(i) + maxArr(i) * GAMMA )
        yBatch.set(yBatchArray)
      }
      Qvalue.foreach(_.dispose())
      actionBatchArr.set(actionBatch.flatten)
      stateBatchArr.set(stateBatch.flatten)

      this.Qnet.forward(Map("data" -> stateBatchArr, "actionInput" -> actionBatchArr), Map("yInput" -> yBatch))
      this.Qnet.backward()
      this.Qnet.update()

      // save network every 1000 iteration
      if (timeStep % 1000 == 0 && saveModelPath != null) {
        println(s"save model: $saveModelPath${File.separator}${"network-dqn_mx%04d.params".format(timeStep)}")
        this.Qnet.saveParams(s"$saveModelPath${File.separator}${"network-dqn_mx%04d.params".format(timeStep)}")
      }
      
      if (timeStep % UPDATE_TIME == 0) copyTargetQNetwork()
    }
    
    def setPerception(nextObservation: Mat, action: Int, reward: Float, terminal: Boolean): Unit = {
      val newState = currentState.drop(screenHeight * screenWidth) ++ dataPrep(nextObservation)
      replayMemory = replayMemory :+ (currentState, action, reward, newState, terminal)
      if (replayMemory.size > REPLAY_MEMORY) {
        replayMemory = replayMemory.drop(1)
      }
      if (timeStep > OBSERVE) {
        // Train the network
        trainQNetwork()
      }

      if (timeStep % 1000 == 0) {
          println(s"TIMESTEP: $timeStep")
      }
      currentState = newState
      timeStep += 1
    }
  }
}
