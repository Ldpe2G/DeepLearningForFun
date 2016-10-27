import java.awt.image.BufferedImage
import java.awt.Graphics2D
import java.awt.Color
import org.opencv.core.Mat
import java.awt.image.DataBufferByte
import org.opencv.core.CvType
import org.opencv.highgui.Highgui
import javax.imageio.ImageIO
import java.io.File
import visual.Imshow
import scala.util.Random
import scala.collection.mutable.Map
import org.opencv.core.Rect

/**
 * @author Depeng Liang
 */
object FlappyBird {
    val FPS = 1000
    val INTERVAL = 1000 / FPS
    val SCREENWIDTH  = 288
    val SCREENHEIGHT = 512
    val PIPEGAPSIZE = 100 // gap between upper and lower part of pipe
    val BASEY = (SCREENHEIGHT * 0.79).toInt

    val PLAYER_WIDTH = Utils.IMAGES("player")(0).getWidth
    val PLAYER_HEIGHT = Utils.IMAGES("player")(0).getHeight
    val PIPE_WIDTH = Utils.IMAGES("pipe")(0).getWidth
    val PIPE_HEIGHT = Utils.IMAGES("pipe")(0).getHeight
    val PLAYER_INDEX_GEN = List(0, 1, 2, 1)
    var playerIdx = 0
    var idx = 0
    var loopIter = 0
    var playerx = (SCREENWIDTH * 0.2).toInt
    var playery = ((SCREENHEIGHT - PLAYER_HEIGHT) / 2)
    var basex = 0
    var score = 0
    val baseShift = Utils.IMAGES("base")(0).getWidth - SCREENWIDTH
    
    var upperPipes: List[Map[String, Int]] = null
    var lowerPipes: List[Map[String, Int]] = null
    
    // player velocity, max velocity, downward accleration, accleration on flap
    val pipeVelX = -4
    var playerVelY = 0  // player's velocity along Y, default same as playerFlapped
    val playerMaxVelY = 10 // max vel along Y, max descend speed
    val playerAccY = 1 // players downward accleration
    val playerFlapAcc = -7 // players speed on flapping
    var playerFlapped = false // True when player flaps
    
    restartGame()
    
    def restartGame(): Unit = {
      idx = 0
      playerx = (SCREENWIDTH * 0.2).toInt
      playery = ((SCREENHEIGHT - PLAYER_HEIGHT) / 2)
      basex = 0
      score = 0
      
      val newPipe1 = getRandomPipe()
      val newPipe2 = getRandomPipe()
      upperPipes = List(
        Map("x" -> SCREENWIDTH, "y" -> newPipe1(0)("y")),
        Map("x" -> (SCREENWIDTH + SCREENWIDTH / 2), "y" -> newPipe2(0)("y"))
      )
     lowerPipes = List(
        Map("x" -> SCREENWIDTH, "y" -> newPipe1(1)("y")),
        Map("x" -> (SCREENWIDTH + SCREENWIDTH / 2), "y" -> newPipe2(1)("y"))
      )

      playerVelY = 0
      playerFlapped = false
    }
    
    val ims = new Imshow("FlappyBird")

    // Constructs a BufferedImage of one of the predefined image types.
    val screen = new BufferedImage(SCREENWIDTH, SCREENHEIGHT, BufferedImage.TYPE_3BYTE_BGR);
    
    // Create a graphics which can be used to draw into the buffered image
    val g2d = screen.createGraphics();
    
    def frameStep(inputAction: Int): (Mat, Float, Boolean) = {
      Thread.sleep(INTERVAL)

      var reward = 0.1f
      var terminal = false
      
      // inputAction == 0: do nothing
      // inputAction == 1: flap the bird
      if (inputAction == 1) {
        if (playery > -2 * PLAYER_HEIGHT) {
          playerVelY = playerFlapAcc
          playerFlapped = true
        }
      }
      
      val playerMidPos = playerx + PLAYER_WIDTH / 2
      for (pipe <- upperPipes) {
        val pipeMidPos = pipe("x") + PIPE_WIDTH / 2
        if (pipeMidPos <= playerMidPos && playerMidPos < pipeMidPos + 4) {
          score += 1
          reward = 1
        }
      }
      
      // playerIndex basex change
      if ((loopIter + 1) % 3 == 0) {
        idx = (idx + 1) % 4
        playerIdx = PLAYER_INDEX_GEN(idx)
      }
      loopIter = (loopIter + 1) % 30
      basex = -((-basex + 100) % baseShift)

      // player's movement
      if (playerVelY < playerMaxVelY && !playerFlapped) playerVelY += playerAccY
      if (playerFlapped) playerFlapped = false
      playery += Math.min(playerVelY, BASEY - playery - PLAYER_HEIGHT)
      if (playery < 0) playery = 0
      
      // move pipes to left
      for ((uPipe, lPipe) <- upperPipes.zip(lowerPipes)) {
        uPipe("x") = uPipe("x") + pipeVelX
        lPipe("x") = lPipe("x") + pipeVelX
      }
      
      // add new pipe when first pipe is about to touch left of screen
      if (0 < upperPipes(0)("x") && upperPipes(0)("x") < 5) {
        val newPipe = getRandomPipe()
        upperPipes :+= newPipe(0)
        lowerPipes :+= newPipe(1)
      }

      // remove first pipe if its out of the screen
      if (upperPipes(0)("x") < -PIPE_WIDTH) {
        upperPipes = upperPipes.drop(1)
        lowerPipes = lowerPipes.drop(1)
      }
      
      // check if crash here
     val isCrash= checkCrash(Map("x" -> playerx, "y" -> playery,
                             "index" -> playerIdx), upperPipes, lowerPipes)

      if (isCrash) {
        terminal = true
        restartGame()
        reward = -1
      }
     
     // draw screen
      g2d.clearRect(0, 0, SCREENWIDTH, SCREENHEIGHT)
      g2d.setColor(Color.black);
      g2d.fillRect(0, 0, SCREENWIDTH, SCREENHEIGHT)
      
      for ((uPipe, lPipe) <- upperPipes.zip(lowerPipes)) {
        g2d.drawImage(Utils.IMAGES("pipe")(0), null, uPipe("x"), uPipe("y"))
        g2d.drawImage(Utils.IMAGES("pipe")(1), null, lPipe("x"), lPipe("y"))
      }
      g2d.drawImage(Utils.IMAGES("base")(0), null, basex, BASEY)
      g2d.drawImage(Utils.IMAGES("player")(playerIdx), null, playerx, playery)
      
      val imageData = Utils.bufferedImageToMat(screen)
      ims.showImage(imageData)
      
      (imageData, reward, terminal)
    }
    
    // returns True if player collders with base or pipes.
    def checkCrash(player: Map[String, Int], upperPipes: List[Map[String, Int]],
        lowerPipes: List[Map[String, Int]]): Boolean = {
      
      val pi = player("index")
      val width = Utils.IMAGES("player")(0).getWidth()
      val height = Utils.IMAGES("player")(0).getHeight()
      var crash = false
      
      // if player crashes into ground
      if (player("y") + height >= BASEY - 1) {
        crash = true
      } else {
        val playerRect = new Rect(player("x"), player("y"), width, height)

        for ((uPipe, lPipe) <- upperPipes.zip(lowerPipes)) {
          // upper and lower pipe rects
          val uPipeRect = new Rect(uPipe("x"), uPipe("y"), PIPE_WIDTH, PIPE_HEIGHT)
          val lPipeRect = new Rect(lPipe("x"), lPipe("y"), PIPE_WIDTH, PIPE_HEIGHT)

          // if bird collided with upipe or lpipe
          val uCollide = pixelCollision(playerRect, uPipeRect, Utils.HITMASKS("player")(pi), Utils.HITMASKS("pipe")(0))
          val lCollide = pixelCollision(playerRect, lPipeRect,  Utils.HITMASKS("player")(pi), Utils.HITMASKS("pipe")(1))

          if (uCollide || lCollide) crash = true
        }
      }
      crash
    }
    
    implicit class MyRect(val rect: Rect) extends AnyVal {
      def clip(otherRect: Rect): Rect = {
        val leftX = Math.max(rect.x, otherRect.x)
        val rightX = Math.min(rect.x + rect.width, otherRect.x + otherRect.width)
        val upY = Math.max(rect.y, otherRect.y)
        val lowY = Math.min(rect.y + rect.height, otherRect.y + otherRect.height)
        if (leftX < rightX && upY < lowY) {
          new Rect(leftX, upY, rightX - leftX, lowY - upY)
        } else null
      }
    }
    
    // Checks if two objects collide and not just their rects
    def pixelCollision(rect1: Rect, rect2: Rect,
        hitmask1: Array[Array[Boolean]], hitmask2: Array[Array[Boolean]]): Boolean = {
      val rect = rect1.clip(rect2)
      
      var collision = false
      if (rect != null) {
        val (x1, y1) = (rect.x - rect1.x, rect.y - rect1.y)
        val (x2, y2) = (rect.x - rect2.x, rect.y - rect2.y)
  
        for (x <- 0 until rect.width) {
          for (y <- 0 until rect.height) {
              if (hitmask1(x1+x)(y1+y) && hitmask2(x2+x)(y2+y)) collision = true
          }
        }
      }
      collision
    }
    
    // returns a randomly generated pipe
    def getRandomPipe(): List[Map[String, Int]] = {
      // y of gap between upper and lower pipe
      val gapYs = List(20, 30, 40, 50, 60, 70, 80, 90)
      val index = Random.nextInt(gapYs.length)
      var gapY = gapYs(index)
  
      gapY += (BASEY * 0.2).toInt
      val pipeX = SCREENWIDTH + 10
  
      List(
          Map("x" -> pipeX, "y" -> (gapY - PIPE_HEIGHT)),  // upper pipe
          Map("x" -> pipeX, "y" -> (gapY + PIPEGAPSIZE))  // lower pipe
      )
    }
    
}