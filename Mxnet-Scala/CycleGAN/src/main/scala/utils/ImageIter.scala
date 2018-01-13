package utils

import org.apache.mxnet.DataIter
import java.io.File
import org.apache.mxnet.NDArray
import scala.collection.immutable.ListMap
import org.apache.mxnet.Shape
import scala.util.Random
import org.apache.mxnet.Context

/**
 * @author Depeng Liang
 */
class ImageIter(dataPath: String, opt: Options.OptTrain, ctx: Context, dataName: String = "data") extends DataIter  {
  override def batchSize: Int = opt.batchSize

  private var imgLists = new File(dataPath).list().toList
  private var idx = 0

  private val imgBuffer = NDArray.empty(Shape(opt.batchSize, opt.inputNC, opt.cropSize, opt.cropSize), ctx)
  
  override def reset(): Unit = {
    imgLists = Random.shuffle(imgLists)
    idx = 0
  }

  override def getData(): IndexedSeq[NDArray] = {
    if (idx < imgLists.length) {
      val imgPath = s"$dataPath/${imgLists(idx)}"
      idx += opt.batchSize
      
      val result = DataProcess.preprocessSingleImage(
          imgPath,
          scaleAndCrop = true,
          scaleSize = opt.scaleSize,
          cropSize = opt.cropSize,
          flip = if (opt.flip == 1) true else false)
      imgBuffer.set(result)
      IndexedSeq(imgBuffer)
    } else throw new NoSuchElementException
  }

  override def hasNext: Boolean = {
    if (idx + opt.batchSize <= imgLists.length) true
    else false
  }

  override def getLabel(): IndexedSeq[NDArray] = IndexedSeq[NDArray]()

  override def getPad(): Int = 0

  override def getIndex(): IndexedSeq[Long] =  IndexedSeq[Long]()

  // The name and shape of data provided by this iterator
  override def provideData: ListMap[String, Shape] =
    ListMap[String, Shape](dataName  -> Shape(opt.batchSize, opt.inputNC, opt.cropSize, opt.cropSize))

  // The name and shape of label provided by this iterator
  override def provideLabel: ListMap[String, Shape] = ListMap[String, Shape]()

}
