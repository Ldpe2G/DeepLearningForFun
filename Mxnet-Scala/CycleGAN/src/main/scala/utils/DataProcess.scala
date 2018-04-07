package utils

//import com.sksamuel.scrimage.Image
//import com.sksamuel.scrimage.Pixel
//import com.sksamuel.scrimage.filter.GaussianBlurFilter
//import com.sksamuel.scrimage.nio.JpegWriter
import org.apache.mxnet.Context
import org.apache.mxnet.NDArray
import java.io.File
import org.apache.mxnet.Shape
import scala.util.Random
import org.opencv.highgui.Highgui
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import org.opencv.core.Size
import org.opencv.core.Rect
import org.opencv.core.Core
import org.opencv.core.CvType
import java.util.Arrays.ArrayList
import java.util.ArrayList

/**
 * @author Depeng Liang
 */
object DataProcess {

  def preprocessSingleImage(path: String, scaleAndCrop: Boolean = false,
      scaleSize: Int = 143, cropSize: Int = 128, flip: Boolean = false): Array[Float] = {
    val img = Highgui.imread(path, 1)
    val newImg = {
      val tmp = if (scaleAndCrop) {
        val scaled = new Mat
        Imgproc.resize(img, scaled, new Size(scaleSize, scaleSize), 0.5, 0.5, Imgproc.INTER_LINEAR)
         val oW, oH = cropSize
         val iH = scaled.height
         val iW = scaled.width
         val h1 = if (iH > oH) Random.nextInt(iH - oH) else 0
         val w1 = if (iW > oW) Random.nextInt(iW - oW) else 0
         val cropped = new Mat(scaled,  new Rect(w1, h1, oW, oH))
        cropped
      } else img
      val flipped = if (flip && Random.nextFloat() > 0.5f) {
        val result = new Mat
        Core.flip(tmp, result, 1)
        result
      } else tmp
      flipped
    }
    newImg.convertTo(newImg, CvType.CV_32F, 1f / 255f)  

    val raw = {
      
      val layers = new ArrayList[Mat]()
      Core.split(newImg, layers)
      val totals = newImg.width() * newImg.height()
      val rA = new Array[Float](totals)
      val gA = new Array[Float](totals)
      val bA = new Array[Float](totals)
      layers.get(2).get(0, 0, rA)
      layers.get(1).get(0, 0, gA)
      layers.get(0).get(0, 0, bA)
      

      rA.map(_ * 2f - 1f) ++ gA.map(_ * 2f - 1f) ++ bA.map(_ * 2f - 1f)
    }
    raw
  }

  def postprocessImage(img: NDArray): Array[Byte] = {
    img.toArray.map(i => (i + 1) / 2f * 255f)
           .map(x => if (x < 0f) 0 else if (x > 255f) 255 else x.toInt)
           .map(_.toByte)
  }
  
    def postprocessImage2(img: Array[Float]): Array[Byte] = {
    img.map(x => if (x < 0f) 0 else if (x > 255f) 255 else x.toInt)
           .map(_.toByte)
  }

  def preprocessSingleImage4Test(path: String, ctx: Context): NDArray = {
    val img = Highgui.imread(path, 1)
    img.convertTo(img, CvType.CV_32F, 1f / 255f)

    val raw = {
      
      val layers = new ArrayList[Mat]()
      Core.split(img, layers)
      val totals = img.width() * img.height()
      val rA = new Array[Float](totals)
      val gA = new Array[Float](totals)
      val bA = new Array[Float](totals)
      layers.get(2).get(0, 0, rA)
      layers.get(1).get(0, 0, gA)
      layers.get(0).get(0, 0, bA)
      
      rA.map(_ * 2f - 1f) ++ gA.map(_ * 2f - 1f) ++ bA.map(_ * 2f - 1f)
    }
    
    val result = NDArray.empty(ctx, 1, 3, img.height(), img.width())
    result.set(raw)
    result
  }
}
