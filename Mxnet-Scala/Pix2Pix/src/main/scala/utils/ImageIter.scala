package utils

import org.apache.mxnet.DataIter
import java.io.File
import org.apache.mxnet.NDArray
import scala.collection.immutable.ListMap
import org.apache.mxnet.Shape
import scala.util.Random
import org.apache.mxnet.Context
import org.apache.mxnet._

/**
 * @author Depeng Liang
 */
class ImageIter(dataPath: String, opt: Options.OptTrain, ctx: Context, dataName: String = "data") extends DataIter  {
  override def batchSize: Int = opt.batchSize

  private var imgLists = new File(dataPath).list().toList
  private var idx = 0

  private val imgABuffer = NDArray.empty(Shape(opt.batchSize, opt.inputNC, opt.cropSize, opt.cropSize), ctx)
  private val imgBBuffer = NDArray.empty(Shape(opt.batchSize, opt.inputNC, opt.cropSize, opt.cropSize), ctx)

  override def reset(): Unit = {
    imgLists = Random.shuffle(imgLists)
    idx = 0
  }

  override def getData(): IndexedSeq[NDArray] = {
    if (idx < imgLists.length) {
      val imgPath = s"$dataPath/${imgLists(idx)}"
      idx += opt.batchSize
      val (resultA, resultB) = DataProcess.preprocessImage(
          imgPath,
          scaleAndCrop = true,
          scaleSize = opt.scaleSize,
          cropSize = opt.cropSize,
          flip = if (opt.flip == 1) true else false)
      if (opt.whichDirection == "AtoB") {
        imgABuffer.set(resultA)
        imgBBuffer.set(resultB)
      } else if (opt.whichDirection == "BtoA") {
        imgBBuffer.set(resultA)
        imgABuffer.set(resultB)
      }
      IndexedSeq(imgABuffer, imgBBuffer)
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

  override def provideDataDesc: IndexedSeq[org.apache.mxnet.DataDesc] =
    IndexedSeq(DataDesc(dataName, Shape(opt.batchSize, opt.inputNC, opt.cropSize, opt.cropSize),
               DType.Float32, Layout.NCHW))

  override def provideLabelDesc: IndexedSeq[org.apache.mxnet.DataDesc] = IndexedSeq[org.apache.mxnet.DataDesc]()
}
