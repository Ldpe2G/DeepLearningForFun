import org.apache.mxnet.Executor
import org.apache.mxnet.NDArray
import org.apache.mxnet.Symbol
import org.apache.mxnet.Shape
import org.apache.mxnet.Normal
import org.apache.mxnet.Context
import org.apache.mxnet.Uniform
import org.apache.mxnet.util.OptionConversion._


object LSTMSymbol {

  case class LSTMModel(exec: Executor, symbol: Symbol, data: NDArray, label: NDArray,
      argsDict: Map[String, NDArray], gradDict: Map[String, NDArray])

  final case class LSTMState(c: Symbol, h: Symbol)
  final case class LSTMParam(i2hWeight: Symbol, i2hBias: Symbol,
                                                         h2hWeight: Symbol, h2hBias: Symbol)

  // LSTM Cell symbol
  private def lstmCell(
    numHidden: Int,
    inData: Symbol,
    prevState: LSTMState,
    param: LSTMParam,
    seqIdx: Int,
    layerIdx: Int,
    dropout: Float = 0f): LSTMState = {
    val inDataa = {
      if (dropout > 0f) Symbol.api.Dropout(inData, p = dropout)
      else inData
    }
    val i2h = Symbol.api.FullyConnected(inDataa,
                                        weight = param.i2hWeight,
                                        bias = param.i2hBias,
                                        num_hidden = numHidden * 4,
                                        name = s"t${seqIdx}_l${layerIdx}_i2h")
    val h2h = Symbol.api.FullyConnected(prevState.h,
                                        weight = param.h2hWeight,
                                        bias = param.h2hBias,
                                        num_hidden = numHidden * 4,
                                        name = s"t${seqIdx}_l${layerIdx}_h2h")
    val gates = i2h + h2h
    val sliceGates = Symbol.api.SliceChannel(gates, num_outputs = 4, name = s"t${seqIdx}_l${layerIdx}_slice")
    val ingate = Symbol.api.sigmoid(sliceGates.get(0))
    val inTransform = Symbol.api.tanh(sliceGates.get(1))
    val forgetGate = Symbol.api.sigmoid(sliceGates.get(2))
    val outGate = Symbol.api.sigmoid(sliceGates.get(3))
    val nextC = (forgetGate * prevState.c) + (ingate * inTransform)
    val nextH = outGate * Symbol.api.tanh(nextC)
    LSTMState(c = nextC, h = nextH)
  }

  private def getSymbol(seqLen: Int, numHidden: Int, numLabel: Int,
      numLstmLayer: Int = 1, dropout: Float = 0f): Symbol = {
    var inputX = Symbol.Variable("data")
    val inputY = Symbol.Variable("softmax_label")

    var paramCells = Array[LSTMParam]()
    var lastStates = Array[LSTMState]()
    for (i <- 0 until numLstmLayer) {
      paramCells = paramCells :+ LSTMParam(i2hWeight = Symbol.Variable(s"l${i}_i2h_weight"),
                                           i2hBias = Symbol.Variable(s"l${i}_i2h_bias"),
                                           h2hWeight = Symbol.Variable(s"l${i}_h2h_weight"),
                                           h2hBias = Symbol.Variable(s"l${i}_h2h_bias"))
      lastStates = lastStates :+ LSTMState(c = Symbol.Variable(s"l${i}_init_c"),
                                           h = Symbol.Variable(s"l${i}_init_h"))
    }
    assert(lastStates.length == numLstmLayer)
    
    val lstmInputs = Symbol.api.SliceChannel(inputX, axis = 1, num_outputs = seqLen, squeeze_axis = true)

    var hiddenAll = Array[Symbol]()
    var dpRatio = 0f
    var hidden: Symbol = null
    for (seqIdx <- 0 until seqLen) {
      hidden = lstmInputs.get(seqIdx)
      // stack LSTM
      for (i <- 0 until numLstmLayer) {
        if (i == 0) dpRatio = 0f else dpRatio = dropout
        val nextState = lstmCell(numHidden, inData = hidden,
                                prevState = lastStates(i),
                                param = paramCells(i),
                                seqIdx = seqIdx, layerIdx = i, dropout = dpRatio)
        hidden = nextState.h
        lastStates(i) = nextState
      }
      //  add dropout before softmax
      if (dropout > 0f) hidden = Symbol.api.Dropout(hidden, p = dropout)
      hiddenAll = hiddenAll :+ hidden
    }

    val finalOut = hiddenAll(hiddenAll.length - 1)
    val fc = Symbol.api.FullyConnected(finalOut, num_hidden = numLabel)
    Symbol.api.SoftmaxOutput(fc, label = inputY)
  }

  def setupModel(seqLen: Int, nInput: Int, numHidden: Int, numLabel: Int, batchSize: Int,
      numLstmLayer: Int = 1, dropout: Float = 0f, ctx: Context = Context.cpu()): LSTMModel = {
    val sym = LSTMSymbol.getSymbol(seqLen, numHidden, numLabel, numLstmLayer = numLstmLayer)
    val argNames = sym.listArguments()
    val auxNames = sym.listAuxiliaryStates()

    val initC = for (l <- 0 until numLstmLayer) yield (s"l${l}_init_c", (batchSize, numHidden))
    val initH = for (l <- 0 until numLstmLayer) yield (s"l${l}_init_h", (batchSize, numHidden))
    val initStates = (initC ++ initH).map(x => x._1 -> Shape(x._2._1, x._2._2)).toMap

    val dataShapes = Map("data" -> Shape(batchSize, seqLen, nInput)) ++ initStates

    val (argShapes, outShapes, auxShapes) = sym.inferShape(dataShapes)

    val initializer = new Uniform(0.1f)
    val argsDict = argNames.zip(argShapes).map { case (name, shape) =>
       val nda = NDArray.zeros(shape, ctx)
       if (!dataShapes.contains(name) && name != "softmax_label") {
         initializer(name, nda)
       }
       name -> nda
    }.toMap

    val argsGradDict = argNames.zip(argShapes)
                               .filter(x => x._1 != "softmax_label" && x._1 != "data")
                               .map( x => x._1 -> NDArray.zeros(x._2, ctx) ).toMap

    val auxDict = auxNames.zip(auxShapes.map(NDArray.zeros(_, ctx))).toMap
    val exec = sym.bind(ctx, argsDict, argsGradDict, "write", auxDict, null, null)

    val data = argsDict("data")
    val label = argsDict("softmax_label")
    
    LSTMModel(exec, sym, data, label, argsDict, argsGradDict)
  }
}