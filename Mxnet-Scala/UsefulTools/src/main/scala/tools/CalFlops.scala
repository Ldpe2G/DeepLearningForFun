package tools

import org.kohsuke.args4j.{CmdLineParser, Option}
import scala.collection.JavaConverters._
import java.lang.System
import scala.util.parsing.json._
import org.kohsuke.args4j.spi.StringArrayOptionHandler
import org.apache.mxnet.Symbol
import org.apache.mxnet.DataDesc
import org.apache.mxnet.Shape
import org.apache.mxnet.Layout
import org.apache.mxnet.module.Module
import scala.collection.IndexedSeq
import org.apache.mxnet.DType

/**
 * @author Depeng Liang
 */
object CalFlops {
  
  def getInternalLabelInfo(internalSym: Symbol,
      labelShapes: scala.Option[IndexedSeq[DataDesc]]): (scala.Option[IndexedSeq[DataDesc]], IndexedSeq[String]) = labelShapes match {
    case None => { (None, null) }
    case Some(lShapes) => {
      val argNames = internalSym.listArguments()
      val internalLabelShapes = lShapes.filter(l => argNames.contains(l.name))
      if (internalLabelShapes.length > 0) {
        (Some(internalLabelShapes), internalLabelShapes.map(_.name))
      } else (None, null)
    }
  }
  
  def numOfBytes(dtype: DType.DType): Int = {
    dtype match {
      case DType.UInt8 => 1
      case DType.Int32 => 4
      case DType.Float16 => 2
      case DType.Float32 => 4
      case DType.Float64 => 8
      case DType.Unknown => 0
    }
  }
  
  def str2Tuple(str: String): List[String] = {
    val re = """\d+""".r
    re.findAllIn(str).toList
  }
  
  def main(args: Array[String]): Unit = {
    val cafl = new CalFlops
    val parser: CmdLineParser = new CmdLineParser(cafl)
    try {
      parser.parseArgument(args.toList.asJava)
      
      assert(cafl.symbol != "")
      assert(cafl.dataShapes.length > 0)
      
      println(s"${cafl.symbol} ${cafl.dataShapes.mkString(", ")} ${cafl.labelShapes.length}")
      
      val network = Symbol.load(cafl.symbol)
      
      val dataShapes = cafl.dataShapes.map { s =>
        val tmp = s.trim().split(",")
        val name = tmp(0)
        val shapes = Shape(tmp.drop(1).map(_.toInt))
        DataDesc(name = name, shape = shapes)
      }.toIndexedSeq
      
      val dataNames = dataShapes.map(_.name)
      
      val (labelShapes, labelNames): (scala.Option[IndexedSeq[DataDesc]], IndexedSeq[String]) = {
        if (cafl.labelShapes.length == 0) {
          (None, null)
        } else {
          val shapes = cafl.labelShapes.map { s =>
            val tmp = s.trim().split(",")
            val name = tmp(0)
            val shapes = Shape(tmp.drop(1).map(_.toInt))
            DataDesc(name = name, shape = shapes)
          }.toIndexedSeq
          val names = shapes.map(_.name)
          (Some(shapes), names)
        }
      }
      
      val argss = network.listArguments()
      assert(labelNames.forall(n => argss.contains(n)))
      
      val module = new Module(network, dataNames, labelNames)
      module.bind(dataShapes, labelShapes, forTraining = false)
      module.initParams()

      val (argParams, auxParams) = module.getParams
      
      
      var totalFlops = 0f
      
      val conf = JSON.parseFull(network.toJson) match {
        case None => null
        case Some(map) => map.asInstanceOf[Map[String, Any]]
      }
      require(conf != null, "Invalid json")
      require(conf.contains("nodes"), "Invalid json")
      val nodes = conf("nodes").asInstanceOf[List[Any]]
      
      nodes.foreach { node =>
        val params = node.asInstanceOf[Map[String, Any]]
        val op = params("op").asInstanceOf[String]
        val name = params("name").asInstanceOf[String]
        val attrs = {
          if (params.contains("attrs")) params("attrs").asInstanceOf[Map[String, String]]
          else if (params.contains("param")) params("param").asInstanceOf[Map[String, String]]
          else Map[String, String]()
        }
        
        val inputs = params("inputs").asInstanceOf[List[List[Double]]]
        
        op match {
          case "Convolution" => {
            val internalSym = network.getInternals().get(name + "_output")
            val (internalLabelShapes, internalLabelNames) = getInternalLabelInfo(internalSym, labelShapes)
            
            val tmpModel = new Module(internalSym, dataNames, internalLabelNames)
            tmpModel.bind(dataShapes, internalLabelShapes, forTraining = false)
            tmpModel.initParams()
            val outShape = tmpModel.getOutputsMerged()(0).shape
            
            // support conv1d NCW and conv2d NCHW layout
            val outShapeProdut = if (outShape.length == 3) outShape(2) else outShape(2) * outShape(3)
            totalFlops += outShapeProdut * argParams(name + "_weight").shape.product * outShape(0)
            
            if (argParams.contains(name + "_bias")) {
              totalFlops += outShape.product
            }
          }
          case "Deconvolution" => {
            val inputLayerName = {
              val inputNode = nodes(inputs(0)(0).toInt).asInstanceOf[Map[String, Any]]
              inputNode("name").asInstanceOf[String]
            }
            
            val internalSym = network.getInternals().get(inputLayerName)
            val (internalLabelShapes, internalLabelNames) = getInternalLabelInfo(internalSym, labelShapes)

            val tmpModel = new Module(internalSym, dataNames, internalLabelNames)
            tmpModel.bind(dataShapes, internalLabelShapes, forTraining = false)
            tmpModel.initParams()
            val inputShape = tmpModel.getOutputsMerged()(0).shape

            totalFlops += inputShape(2) * inputShape(3) * argParams(name + "_weight").shape.product * inputShape(0)
            
            if (argParams.contains(name + "_bias")) {
              val internalSym = network.getInternals().get(name + "_output")
              val (internalLabelShapes, internalLabelNames) = getInternalLabelInfo(internalSym, labelShapes)
              
              val tmpModel = new Module(internalSym, dataNames, internalLabelNames)
              tmpModel.bind(dataShapes, internalLabelShapes, forTraining = false)
              tmpModel.initParams()
              val outShape = tmpModel.getOutputsMerged()(0).shape
              
              totalFlops += outShape.product
            }
          }
          case "FullyConnected" => {
            totalFlops += argParams(name + "_weight").shape.product * dataShapes(0).shape(0)
            if (argParams.contains(name + "_bias")) {
              val numFilter = argParams(name + "_bias").shape(0)
              totalFlops += numFilter * dataShapes(0).shape(0)
            }
          }

          case "Pooling" => {
            val globalPool = {
              if (!attrs.contains("global_pool")) false
              else if (attrs("global_pool") == "False") false
              else true
            }
            if (globalPool) {
              val inputLayerName = {
                val inputNode = nodes(inputs(0)(0).toInt).asInstanceOf[Map[String, Any]]
                inputNode("name").asInstanceOf[String]
              }

              val internalSym = network.getInternals().get(inputLayerName + "_output")
              val (internalLabelShapes, internalLabelNames) = getInternalLabelInfo(internalSym, labelShapes)

              val tmpModel = new Module(internalSym, dataNames, internalLabelNames)
              tmpModel.bind(dataShapes, internalLabelShapes, forTraining = false)
              tmpModel.initParams()
              val inputShape = tmpModel.getOutputsMerged()(0).shape
  
              totalFlops += inputShape.product
            } else {
              val internalSym = network.getInternals().get(name + "_output")
              val (internalLabelShapes, internalLabelNames) = getInternalLabelInfo(internalSym, labelShapes)
              
              val tmpModel = new Module(internalSym, dataNames, internalLabelNames)
              tmpModel.bind(dataShapes, internalLabelShapes, forTraining = false)
              tmpModel.initParams()
              val outShape = tmpModel.getOutputsMerged()(0).shape
              val kernelP = str2Tuple(attrs("kernel")).map(_.toInt).reduce(_ * _)
              
              totalFlops += outShape.product * kernelP
            }
          }
          case "BatchNorm" => {}
          case "Activation" => {
            attrs("act_type") match {
              case "relu" => {
                val inputLayerName = {
                  val inputNode = nodes(inputs(0)(0).toInt).asInstanceOf[Map[String, Any]]
                  inputNode("name").asInstanceOf[String]
                }
  
                val internalSym = network.getInternals().get(inputLayerName + "_output")
                val (internalLabelShapes, internalLabelNames) = getInternalLabelInfo(internalSym, labelShapes)
  
                val tmpModel = new Module(internalSym, dataNames, internalLabelNames)
                tmpModel.bind(dataShapes, internalLabelShapes, forTraining = false)
                tmpModel.initParams()
                val inputShape = tmpModel.getOutputsMerged()(0).shape
                
                totalFlops += inputShape.product
              }
              case _ => {}
            }
          }
          case "LeakyReLU" => {}
          case _ => {}
        }
      }
      
      val totalSize = {
        val argSize = (0f /: argParams){ (size, elem) =>
          size + elem._2.shape.product * numOfBytes(elem._2.dtype)
        }
        val auxSize = (0f /: auxParams){ (size, elem) =>
          size + elem._2.shape.product * numOfBytes(elem._2.dtype)
        }
        argSize + auxSize
      }


    println(s"flops: ${totalFlops / 1000000} MFLOPS")
    println(s"model size: ${totalSize / 1024 / 1024} MB")
      
    } catch {
      case ex: Exception => {
        println(ex.getMessage, ex)
        parser.printUsage(System.err)
        System.exit(1)
      }
    }
  }
}

class CalFlops {
  @Option(name = "--ds", handler = classOf[StringArrayOptionHandler],
      required = true, usage = "the network json file to calculate flops.")
  private val dataShapes: Array[String] = Array[String]()
  @Option(name = "--ls", handler = classOf[StringArrayOptionHandler],
      usage = "the network json file to calculate flops.")
  private val labelShapes: Array[String] = Array[String]()
  @Option(name = "--symbol", usage = "the network json file to calculate flops.")
  private val symbol: String = ""
}

