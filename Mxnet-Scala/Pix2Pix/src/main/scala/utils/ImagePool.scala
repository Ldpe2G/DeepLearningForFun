package utils

import org.apache.mxnet.NDArray
import scala.util.Random

class ImagePool(poolSize: Int) {
  require(poolSize > 0)

  val images = scala.collection.mutable.Map[Int, NDArray]()
  var numImgs = 0

  def query(image: NDArray): NDArray = {
    if (this.numImgs < this.poolSize) {
      this.images += this.numImgs -> image.copy()
      this.numImgs += 1
      image
    } else {
      val p = Random.nextFloat()
      if (p > 0.5f) {
        val randomId = Random.nextInt(this.poolSize)
        val tmp = this.images(randomId)
        this.images(randomId) = image.copy()
        tmp
      } else image
    }
  }

}
