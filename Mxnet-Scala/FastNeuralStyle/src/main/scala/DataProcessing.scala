
import com.sksamuel.scrimage.Image
import com.sksamuel.scrimage.Pixel
import com.sksamuel.scrimage.filter.GaussianBlurFilter
import com.sksamuel.scrimage.nio.JpegWriter
import org.apache.mxnet.Context
import org.apache.mxnet.NDArray
import java.io.File
import org.apache.mxnet.Shape
import scala.util.Random
import org.opencv.core.Core
import org.opencv.highgui.Highgui
import org.opencv.imgproc.Imgproc
import org.apache.mxnet.NDArray
import org.opencv.core.Mat
import org.opencv.core.CvType
import java.util.ArrayList
import org.opencv.core.Size

/**
 * @author Depeng Liang
 */
object DataProcessing {
  
  nu.pattern.OpenCV.loadShared()

  def preprocessContentImage(path: String,
      dShape: Shape = null, ctx: Context): NDArray = {
    val img = Image(new File(path))
    val resizedImg = img.scaleTo(dShape(3), dShape(2))
    val sample = NDArray.empty(Shape(1, 3, resizedImg.height, resizedImg.width), ctx)
    val datas = {
      val rgbs = resizedImg.iterator.toArray.map { p =>
        (p.red, p.green, p.blue)
      }
      val r = rgbs.map(_._1 - 123.68f)
      val g = rgbs.map(_._2 - 116.779f)
      val b = rgbs.map(_._3 - 103.939f)
      r ++ g ++ b
    }
    sample.set(datas)
    sample
  }

  def preprocessStyleImage(path: String, shape: Shape, ctx: Context): NDArray = {
    val img = Image(new File(path))
    val resizedImg = img.scaleTo(shape(3), shape(2))
    val sample = NDArray.empty(Shape(1, 3, shape(2), shape(3)), ctx)
    val datas = {
      val rgbs = resizedImg.iterator.toArray.map { p =>
        (p.red, p.green, p.blue)
      }
      val r = rgbs.map(_._1 - 123.68f)
      val g = rgbs.map(_._2 - 116.779f)
      val b = rgbs.map(_._3 - 103.939f)
      r ++ g ++ b
    }
    sample.set(datas)
    sample
  }

  def clip(array: Array[Float]): Array[Float] = array.map { a =>
    if (a < 0) 0f
    else if (a > 255) 255f
    else a
  }

  def postprocessImage(img: NDArray): Image = {
    val datas = img.toArray
    val spatialSize = img.shape(2) * img.shape(3)
    val r = clip(datas.take(spatialSize).map(_ + 123.68f))
    val g = clip(datas.drop(spatialSize).take(spatialSize).map(_ + 116.779f))
    val b = clip(datas.takeRight(spatialSize).map(_ + 103.939f))
    val pixels = for (i <- 0 until spatialSize)
      yield Pixel(r(i).toInt, g(i).toInt, b(i).toInt, 255)
    Image(img.shape(3), img.shape(2), pixels.toArray)
  }
  
  def imageToMat(image: Image): Mat = {
    val rgbs = image.iterator.toArray.map { p =>
        (p.red, p.green, p.blue)
      }
    
    val rA = rgbs.map(_._1.toByte)
    val gA = rgbs.map(_._2.toByte)
    val bA = rgbs.map(_._3.toByte)
    
    val rr = new Mat(image.height, image.width, CvType.CV_8U)
    rr.put(0, 0, rA)
    val gg = new Mat(image.height, image.width, CvType.CV_8U)
    gg.put(0, 0, gA)
    val bb = new Mat(image.height, image.width, CvType.CV_8U)
    bb.put(0, 0, bA)

    val result = new Mat()
    val layers = new ArrayList[Mat]()
    layers.add(bb)
    layers.add(gg)
    layers.add(rr)
    Core.merge(layers, result)
    result
  }

  def saveImage(out: Image, filename: String, radius: Int): Unit = {
//    val out = postprocessImage(img)
//    val gauss = GaussianBlurFilter(radius).op
//    val result = Image(out.width, out.height)
//    gauss.filter(out.awt, result.awt)
    implicit val writer = JpegWriter().withCompression(10)
    out.output(filename)
  }
}
