import java.awt.image.BufferedImage
import javax.imageio.ImageIO
import java.io.File
import java.awt.geom.AffineTransform
import org.opencv.core.Mat
import org.opencv.core.CvType
import java.awt.image.DataBufferByte
import scala.collection.mutable.ArrayBuffer
import java.awt.Color

/**
 * @author Depeng Liang
 */
object Utils {
  val root = FlappyBirdDQN.root
  
  var IMAGES = Map[String, List[BufferedImage]]()
  
  IMAGES += "player" -> List(
    ImageIO.read(new File(s"$root/sprites/redbird-upflap.png")),
    ImageIO.read(new File(s"$root/sprites/redbird-midflap.png")),
    ImageIO.read(new File(s"$root/sprites/redbird-downflap.png"))
  )
  
  IMAGES += "pipe" -> List(
    createRotated(ImageIO.read(new File(s"$root/sprites/pipe-green.png"))),
    ImageIO.read(new File(s"$root/sprites/pipe-green.png"))
  )
  
  IMAGES += "base" -> List(ImageIO.read(new File(s"$root/sprites/base.png")))

  // hismask for pipes
  var HITMASKS = Map[String, List[Array[Array[Boolean]]]]()

  HITMASKS += "pipe" -> List(
    getHitmask(IMAGES("pipe")(0)),
    getHitmask(IMAGES("pipe")(1))
  )
  
  // hitmask for player
  HITMASKS += "player" -> List(
    getHitmask(IMAGES("player")(0)),
    getHitmask(IMAGES("player")(1)),
    getHitmask(IMAGES("player")(2))
  )

  //  returns a hitmask using an image's alpha.
  def getHitmask(image: BufferedImage): Array[Array[Boolean]] = {
    val mask = ArrayBuffer[Array[Boolean]]()
    for (x <- 0 until image.getWidth()) {
      val tmp = ArrayBuffer[Boolean]()
      for (y <- 0 until image.getHeight()) {
        val color = new Color(image .getRGB(x, y), true)
        tmp += (if (color.getAlpha == 0) false else true)
      }
      mask += tmp.toArray
    }
    mask.toArray
  }

  def createRotated(image: BufferedImage): BufferedImage = {
    val at = AffineTransform.getRotateInstance(Math.PI, image.getWidth() / 2, image.getHeight() / 2.0)
    createTransformed(image, at)
  }
  
  def createTransformed(image: BufferedImage, at: AffineTransform): BufferedImage = {
    val newImage = new BufferedImage(image.getWidth(), image.getHeight(), BufferedImage.TYPE_INT_ARGB)
    val g = newImage.createGraphics()
    g.transform(at)
    g.drawImage(image, 0, 0, null)
    g.dispose()
    newImage
  }
  
  def bufferedImageToMat(bi: BufferedImage): Mat =  {
    val mat = new Mat(bi.getHeight(), bi.getWidth(), CvType.CV_8UC3)
    val data = (bi.getRaster().getDataBuffer().asInstanceOf[DataBufferByte]).getData();
    mat.put(0, 0, data);
    mat
  }
  
}