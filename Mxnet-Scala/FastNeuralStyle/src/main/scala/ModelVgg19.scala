
import org.apache.mxnet.Executor
import org.apache.mxnet.NDArray
import org.apache.mxnet.Symbol
import org.apache.mxnet.Context
import org.apache.mxnet.Shape
import org.apache.mxnet.util.OptionConversion._

/**
 * @author Depeng Liang
 */
object ModelVgg19 {
  case class ConvExecutor(executor: Executor, data: NDArray, dataGrad: NDArray,
                      style: Array[NDArray], content: NDArray, argDict: Map[String, NDArray])

  def getVggSymbol(prefix: String, contentOnly: Boolean = false): (Symbol, Symbol) = {
    // declare symbol
    val data = Symbol.Variable(s"${prefix}_data")
    val conv1_1 = Symbol.api.Convolution(data, num_filter = 64, pad = Shape(1,1),
                                        kernel = Shape(3,3), stride = Shape(1,1),
                                        no_bias = false, workspace = 1024, name = s"${prefix}_conv1_1")                        
    val relu1_1 = Symbol.api.relu(conv1_1, name = s"${prefix}_relu1_1")                                  
    val conv1_2 = Symbol.api.Convolution(relu1_1, num_filter = 64, pad = Shape(1,1),
                                         kernel = Shape(3,3), stride = Shape(1,1), no_bias = false,
                                         workspace = 1024, name = s"${prefix}_conv1_2")
    val relu1_2 = Symbol.api.relu(conv1_2, name = s"${prefix}_relu1_2")
    val pool1 = Symbol.api.Pooling(relu1_2 , pad = Shape(0,0), kernel = Shape(2,2),
                                   stride = Shape(2,2), pool_type = "avg", name = s"${prefix}_pool1")
    val conv2_1 = Symbol.api.Convolution(pool1, num_filter = 128, pad = Shape(1,1),
                                         kernel = Shape(3,3), stride = Shape(1,1),
                                         no_bias = false, workspace = 1024, name = s"${prefix}_conv2_1")
    val relu2_1 = Symbol.api.relu(conv2_1, name = s"${prefix}_relu2_1")                                     
    val conv2_2 = Symbol.api.Convolution(relu2_1, num_filter = 128, pad = Shape(1,1),
                                          kernel = Shape(3,3), stride = Shape(1,1),
                                          no_bias = false, workspace = 1024, name = s"${prefix}_conv2_2")
    val relu2_2 = Symbol.api.relu(conv2_2, name = s"${prefix}_relu2_2")                                      
    val pool2 = Symbol.api.Pooling(relu2_2, pad = Shape(0, 0), kernel = Shape(2, 2), stride = Shape(2, 2), pool_type = "avg", name = "pool2")
    val conv3_1 = Symbol.api.Convolution(pool2, num_filter =  256, pad = Shape(1,1), kernel  = Shape(3,3), stride = Shape(1,1), no_bias = false, workspace = 1024, name = s"${prefix}_conv3_1")
    val relu3_1 = Symbol.api.relu(conv3_1, name = s"${prefix}_relu3_1")
    val conv3_2 = Symbol.api.Convolution(relu3_1, num_filter = 256, pad = Shape(1,1), kernel = Shape(3,3), stride = Shape(1,1), no_bias =  false, workspace = 1024)
    val relu3_2 = Symbol.api.relu(conv3_2, name = s"${prefix}_relu3_2")
    val conv3_3 = Symbol.api.Convolution(relu3_2, num_filter = 256, pad = Shape(1,1), kernel = Shape(3,3), stride = Shape(1,1), no_bias = false, workspace = 1024)    
    val relu3_3 = Symbol.api.relu(conv3_3, name = s"${prefix}_relu3_3")    
    val conv3_4 = Symbol.api.Convolution(relu3_3, num_filter = 256, pad = Shape(1,1), kernel = Shape(3,3), stride = Shape(1,1), no_bias = false, workspace = 1024)    
    val relu3_4 = Symbol.api.relu(conv3_4 , name = s"${prefix}_relu3_4")    
    val pool3 = Symbol.api.Pooling(relu3_4, pad = Shape(0,0), kernel = Shape(2,2), stride = Shape(2,2), pool_type = "avg", name = s"${prefix}_pool3")
    val conv4_1 = Symbol.api.Convolution(pool3, num_filter = 512, pad = Shape(1,1), kernel = Shape(3,3), stride = Shape(1,1), no_bias = false, workspace = 1024, name = s"${prefix}_conv4_1")
    val relu4_1 = Symbol.api.relu(conv4_1, name = s"${prefix}_relu4_1")
    val conv4_2 = Symbol.api.Convolution(relu4_1, num_filter = 512, pad = Shape(1,1), kernel = Shape(3,3), stride = Shape(1,1), no_bias = false, workspace = 1024, name = s"${prefix}_conv4_2")
    val relu4_2 = Symbol.api.relu(conv4_2, name = s"${prefix}_relu4_2")
    val conv4_3 = Symbol.api.Convolution(relu4_2, num_filter = 512, pad = Shape(1,1), kernel = Shape(3,3), stride = Shape(1,1), no_bias = false, workspace = 1024, name = s"${prefix}_conv4_3")
    val relu4_3 = Symbol.api.relu(conv4_3, name = s"${prefix}_relu4_3")    
    val conv4_4 = Symbol.api.Convolution(relu4_3, num_filter = 512, pad = Shape(1,1), kernel = Shape(3,3), stride = Shape(1,1), no_bias = false, workspace = 1024, name = s"${prefix}_conv4_4")
   val relu4_4 = Symbol.api.relu(conv4_4, name = s"${prefix}_relu4_4")   
   val pool4 = Symbol.api.Pooling(relu4_4, pad = Shape(0,0), kernel = Shape(2,2), stride = Shape(2,2), pool_type = "avg", name = s"${prefix}_pool4")
   val conv5_1 = Symbol.api.Convolution(pool4, num_filter = 512, pad = Shape(1,1), kernel = Shape(3,3), stride = Shape(1,1), no_bias = false, workspace = 1024, name = s"${prefix}_conv5_1")
   val relu5_1 = Symbol.api.relu(conv5_1, name = s"${prefix}_relu5_1")

    // style and content layers
    val style = if (contentOnly) null else Symbol.Group(relu1_2, relu2_2, relu3_3, relu4_3)
    val content = Symbol.Group(relu2_2)
    (style, content)
  }
}
