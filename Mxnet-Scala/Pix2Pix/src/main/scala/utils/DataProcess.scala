package utils

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

  def preprocessImage(path: String, scaleAndCrop: Boolean = true,
      scaleSize: Int = 286, cropSize: Int = 256, flip: Boolean = true): (Array[Float], Array[Float]) = {
    val img = Highgui.imread(path, 1)
    val imA = new Mat(img,  new Rect(0, 0, img.width() / 2, img.height()))
    val imB = new Mat(img,  new Rect(img.width() / 2, 0, img.width() / 2, img.height()))
    val (newImgA, newImgB) = {
      val (tmpA, tmpB) =  if (scaleAndCrop) {
        (reScaleAndCrop(imA, scaleSize, cropSize), reScaleAndCrop(imB, scaleSize, cropSize))
      } else (imA, imB)
      if (flip && Random.nextFloat() > 0.5f) {
        Core.flip(tmpA, tmpA, 1)
        Core.flip(tmpB, tmpB, 1)
      }
      (tmpA, tmpB)
    }
   
    val rawA = {
      val layers = new ArrayList[Mat]()
      Core.split(newImgA, layers)
      val totals = newImgA.width() * newImgA.height()
      val rA = new Array[Float](totals)
      val gA = new Array[Float](totals)
      val bA = new Array[Float](totals)
      layers.get(2).get(0, 0, rA)
      layers.get(1).get(0, 0, gA)
      layers.get(0).get(0, 0, bA)
      rA.map(_ * 2f - 1f) ++ gA.map(_ * 2f - 1f) ++ bA.map(_ * 2f - 1f)
    }
    val rawB = {
      val layers = new ArrayList[Mat]()
      Core.split(newImgB, layers)
      val totals = newImgB.width() * newImgB.height()
      val rA = new Array[Float](totals)
      val gA = new Array[Float](totals)
      val bA = new Array[Float](totals)
      layers.get(2).get(0, 0, rA)
      layers.get(1).get(0, 0, gA)
      layers.get(0).get(0, 0, bA)
      rA.map(_ * 2f - 1f) ++ gA.map(_ * 2f - 1f) ++ bA.map(_ * 2f - 1f)
    }
    (rawA, rawB)
  }

  def reScaleAndCrop(img: Mat, scaleSize: Int, cropSize: Int): Mat = {
    val newImg = {
      val scaled = new Mat
      Imgproc.resize(img, scaled, new Size(scaleSize, scaleSize), 0.5, 0.5, Imgproc.INTER_LINEAR)
      val oW, oH = cropSize
      val iH = scaled.height
      val iW = scaled.width
      val h1 = if (iH > oH) Random.nextInt(iH - oH) else 0
      val w1 = if (iW > oW) Random.nextInt(iW - oW) else 0
      val cropped = new Mat(scaled,  new Rect(w1, h1, oW, oH))
      cropped
    }
    newImg.convertTo(newImg, CvType.CV_32F, 1f / 255f)
    newImg
  }

  def postprocessImage(img: NDArray): Array[Byte] = {
    img.toArray.map(i => (i + 1) / 2f * 255f)
           .map(x => if (x < 0f) 0 else if (x > 255f) 255 else x.toInt)
           .map(_.toByte)
  }

  def preprocessSingleImage4Test(path: String, ctx: Context, direction: String = "AtoB"): NDArray = {
    val img = Highgui.imread(path, 1)
    img.convertTo(img, CvType.CV_32F, 1f / 255f)
    val imA = new Mat(img,  new Rect(0, 0, img.width() / 2, img.height()))
    val imB = new Mat(img,  new Rect(img.width() / 2, 0, img.width() / 2, img.height()))
    
    val imgg = if (direction == "AtoB") imA else imB
    
    val raw = {
      val layers = new ArrayList[Mat]()
      Core.split(imgg, layers)
      val totals = imgg.width() * imgg.height()
      val rA = new Array[Float](totals)
      val gA = new Array[Float](totals)
      val bA = new Array[Float](totals)
      layers.get(2).get(0, 0, rA)
      layers.get(1).get(0, 0, gA)
      layers.get(0).get(0, 0, bA)

      rA.map(_ * 2f - 1f) ++ gA.map(_ * 2f - 1f) ++ bA.map(_ * 2f - 1f)
    }

    val result = NDArray.empty(ctx, 1, 3, imgg.height(), imgg.width())
    result.set(raw)
    result
  }
}
