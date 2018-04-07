package snns

import org.apache.mxnet.{Symbol => Sym}
import org.apache.mxnet.Shape

object Models {
  def getBnVGG16Sym(numClasses: Int = 10): Sym = {
    val data = Sym.Variable(name = "data")
    
    // feature layers one
    var internel_layer = Sym.Convolution("conv1_1")()(Map("data" -> data, "num_filter" -> 64, "kernel" -> "(3, 3)", "pad" -> "(1, 1)"))
    internel_layer = Sym.BatchNorm("bn1_1")()(Map("data" -> internel_layer))
    internel_layer = Sym.Activation("relu1_1")()(Map("data" -> internel_layer, "act_type" -> "relu"))
    
    internel_layer = Sym.Convolution("conv1_2")()(Map("data" -> internel_layer, "num_filter" -> 64, "kernel" -> "(3, 3)", "pad" -> "(1, 1)"))
    internel_layer = Sym.BatchNorm("bn1_2")()(Map("data" -> internel_layer))
    internel_layer = Sym.Activation("relu1_2")()(Map("data" -> internel_layer, "act_type" -> "relu"))
    
    internel_layer = Sym.Pooling("pool1")()(Map("data" ->internel_layer, "pool_type" -> "max", "kernel" -> "(2, 2)", "stride" -> "(2,2)"))
    
    // feature layers two
    internel_layer = Sym.Convolution("conv2_1")()(Map("data" -> internel_layer, "num_filter" -> 128, "kernel" -> "(3, 3)", "pad" -> "(1, 1)"))
    internel_layer = Sym.BatchNorm("bn2_1")()(Map("data" -> internel_layer))
    internel_layer = Sym.Activation("relu2_1")()(Map("data" -> internel_layer, "act_type" -> "relu"))
    
    internel_layer = Sym.Convolution("conv2_2")()(Map("data" -> internel_layer, "num_filter" -> 128, "kernel" -> "(3, 3)", "pad" -> "(1, 1)"))
    internel_layer = Sym.BatchNorm("bn2_2")()(Map("data" -> internel_layer))
    internel_layer = Sym.Activation("relu2_2")()(Map("data" -> internel_layer, "act_type" -> "relu"))
    
    internel_layer = Sym.Pooling("pool2")()(Map("data" ->internel_layer, "pool_type" -> "max", "kernel" -> "(2, 2)", "stride" -> "(2,2)"))
    
    // feature layers three
    internel_layer = Sym.Convolution("conv3_1")()(Map("data" -> internel_layer, "num_filter" -> 256, "kernel" -> "(3, 3)", "pad" -> "(1, 1)"))
    internel_layer = Sym.BatchNorm("bn3_1")()(Map("data" -> internel_layer))
    internel_layer = Sym.Activation("relu3_1")()(Map("data" -> internel_layer, "act_type" -> "relu"))
    
    internel_layer = Sym.Convolution("conv3_2")()(Map("data" -> internel_layer, "num_filter" -> 256, "kernel" -> "(3, 3)", "pad" -> "(1, 1)"))
    internel_layer = Sym.BatchNorm("bn3_2")()(Map("data" -> internel_layer))
    internel_layer = Sym.Activation("relu3_2")()(Map("data" -> internel_layer, "act_type" -> "relu"))
    
    internel_layer = Sym.Convolution("conv3_3")()(Map("data" -> internel_layer, "num_filter" -> 256, "kernel" -> "(3, 3)", "pad" -> "(1, 1)"))
    internel_layer = Sym.BatchNorm("bn3_3")()(Map("data" -> internel_layer))
    internel_layer = Sym.Activation("relu3_3")()(Map("data" -> internel_layer, "act_type" -> "relu"))
    
    internel_layer = Sym.Pooling("pool3")()(Map("data" ->internel_layer, "pool_type" -> "max", "kernel" -> "(2, 2)", "stride" -> "(2,2)"))
    
    // feature layers four
    internel_layer = Sym.Convolution("conv4_1")()(Map("data" -> internel_layer, "num_filter" -> 512, "kernel" -> "(3, 3)", "pad" -> "(1, 1)"))
    internel_layer = Sym.BatchNorm("bn4_1")()(Map("data" -> internel_layer))
    internel_layer = Sym.Activation("relu4_1")()(Map("data" -> internel_layer, "act_type" -> "relu"))
    
    internel_layer = Sym.Convolution("conv4_2")()(Map("data" -> internel_layer, "num_filter" -> 512, "kernel" -> "(3, 3)", "pad" -> "(1, 1)"))
    internel_layer = Sym.BatchNorm("bn4_2")()(Map("data" -> internel_layer))
    internel_layer = Sym.Activation("relu4_2")()(Map("data" -> internel_layer, "act_type" -> "relu"))
    
    internel_layer = Sym.Convolution("conv4_3")()(Map("data" -> internel_layer, "num_filter" -> 512, "kernel" -> "(3, 3)", "pad" -> "(1, 1)"))
    internel_layer = Sym.BatchNorm("bn4_3")()(Map("data" -> internel_layer))
    internel_layer = Sym.Activation("relu4_3")()(Map("data" -> internel_layer, "act_type" -> "relu"))
    
    internel_layer = Sym.Pooling("pool4")()(Map("data" ->internel_layer, "pool_type" -> "max", "kernel" -> "(2, 2)", "stride" -> "(2,2)"))
    
    // feature layers five
    internel_layer = Sym.Convolution("conv5_1")()(Map("data" -> internel_layer, "num_filter" -> 512, "kernel" -> "(3, 3)", "pad" -> "(1, 1)"))
    internel_layer = Sym.BatchNorm("bn5_1")()(Map("data" -> internel_layer))
    internel_layer = Sym.Activation("relu5_1")()(Map("data" -> internel_layer, "act_type" -> "relu"))
    
    internel_layer = Sym.Convolution("conv5_2")()(Map("data" -> internel_layer, "num_filter" -> 512, "kernel" -> "(3, 3)", "pad" -> "(1, 1)"))
    internel_layer = Sym.BatchNorm("bn5_2")()(Map("data" -> internel_layer))
    internel_layer = Sym.Activation("relu5_2")()(Map("data" -> internel_layer, "act_type" -> "relu"))
    
    internel_layer = Sym.Convolution("conv5_3")()(Map("data" -> internel_layer, "num_filter" -> 512, "kernel" -> "(3, 3)", "pad" -> "(1, 1)"))
    internel_layer = Sym.BatchNorm("bn5_3")()(Map("data" -> internel_layer))
    internel_layer = Sym.Activation("relu5_3")()(Map("data" -> internel_layer, "act_type" -> "relu"))
    
    internel_layer = Sym.Pooling("pool5")()(Map("data" -> internel_layer, "pool_type" -> "max", "kernel" -> "(3, 3)", "global_pool" -> true))
    
    // classification layer
    val fc6 = Sym.Convolution("fc6")()(Map("data" -> internel_layer, "num_filter" -> 4096, "kernel" -> "(1, 1)"))
    val bn6 = Sym.BatchNorm("bn6")()(Map("data" -> fc6))
    val relu6 = Sym.Activation("relu6")()(Map("data" -> bn6, "act_type" -> "relu"))    
    val drop6 = Sym.Dropout("drop6")()(Map("data" -> relu6, "p" -> 0.5))
        
    val fc7 = Sym.Convolution("fc7")()(Map("data" -> drop6, "num_filter" -> 4096, "kernel" -> "(1, 1)"))
    val bn7 = Sym.BatchNorm("bn7")()(Map("data" -> fc7))
    val relu7 = Sym.Activation("relu7")()(Map("data" -> bn7, "act_type" -> "relu"))    
    val drop7 = Sym.Dropout("drop7")()(Map("data" -> relu7, "p" -> 0.5))
      
    val fc8 = Sym.Convolution("fc8")()(Map("data" -> drop7, "kernel" -> "(1, 1)", "num_filter" -> numClasses))
    val flatten = Sym.Flatten("flatten")()(Map("data" -> fc8))
    val softmax = Sym.SoftmaxOutput("softmax")()(Map("data" -> flatten))
    softmax
  }
  
  def main(args: Array[String]): Unit = {
    val vgg16Sym = getBnVGG16Sym(numClasses = 10)
    val inShape = Shape(1, 3, 32, 32)
    
    val (_, outShapes, _) = vgg16Sym.inferShape(Map("data" -> inShape))
    
    outShapes.foreach(s => println(s"outShape: ${s}"))
    
  }
  
}




