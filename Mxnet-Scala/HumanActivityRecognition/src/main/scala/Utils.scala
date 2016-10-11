import scala.io.Source


object Utils {

  // Useful Constants

  // Those are separate normalised input features for the neural network
  private val INPUT_SIGNAL_TYPES = Array(
    "body_acc_x_",
    "body_acc_y_",
    "body_acc_z_",
    "body_gyro_x_",
    "body_gyro_y_",
    "body_gyro_z_",
    "total_acc_x_",
    "total_acc_y_",
    "total_acc_z_"
  )

  // Output classes to learn how to classify
  private val LABELS = Array(
    "WALKING", 
    "WALKING_UPSTAIRS", 
    "WALKING_DOWNSTAIRS", 
    "SITTING", 
    "STANDING", 
    "LAYING"
  )

  def loadDatas(dataPath: String, name: String): Array[Array[Array[Float]]] = {
    val dataSignalsPaths = INPUT_SIGNAL_TYPES.map( signal => s"$dataPath/${signal}${name}.txt" )
    val signals = dataSignalsPaths.map { path => 
      Source.fromFile(path).mkString.split("\n").map { line => 
        line.replaceAll("  ", " ").trim().split(" ").map(_.toFloat) }
    }

    val inputDim = signals.length
    val numSamples = signals(0).length
    val timeStep = signals(0)(0).length  

    (0 until numSamples).map { n => 
      (0 until timeStep).map { t =>
        (0 until inputDim).map( i => signals(i)(n)(t) ).toArray
      }.toArray
    }.toArray
  }

  def loadLabel(labelPath: String): Array[Float] = {
    Source.fromFile(labelPath).mkString.split("\n").map(_.toFloat - 1)
  }

}
