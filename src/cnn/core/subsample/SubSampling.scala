package cnn.core.subsample


import cnn.core.objects.Rectangle
import scala.annotation.tailrec
import cnn.exceptions.PoolingException
import cnn.core.structure.{NonEmptyMat, Mat, EmptyMat}
import cnn.helper.implicits.Implicits._
import cnn.exceptions.PoolingException
import cnn.core.structure.NonEmptyMat
import cnn.exceptions.{POOLING_MAX_NEIGHBOR, POOLING_UPSAMPLE_MAT_SIZE}



object SubSampling {
  
  implicit class Pooling(s : NonEmptyMat){
    
    val sWidth = s.width
    val sHeigh = s.heigh
    
	  def max(neighborhood : Tuple2[Int, Int]): NonEmptyMat = {
      if(!s.%(neighborhood._1, neighborhood._2))
        throw PoolingException(POOLING_MAX_NEIGHBOR)
      compute(Option.apply(Rectangle(0,0, neighborhood._1, neighborhood._2)))
    }
  
    /**
     * Upsample the matrix and allow to propagate the error through the pooling layer
     * - the downSampled parameter gives the downsampled matrix and it's corresponding error matrix
     *   the first matrix of the tuple is the downsampled and the second is the error
     *   **/
    def upsample(windowX : Int, windowY : Int, downSampled : NonEmptyMat, error : NonEmptyMat) : NonEmptyMat = {
      
      @tailrec
      def propagateError(rect : Option[Rectangle], acc : NonEmptyMat ,downSampledX : Int, downSampledY : Int) : NonEmptyMat = rect match {
        
         case None => acc
         case r@ Some(_) =>
             val i = downSampled(downSampledX,downSampledY)
             val res =  for(x <- r.get.x to r.get.width-1)
                          yield for(y <- r.get.y to r.get.heigh-1)
                           yield s(x,y) match{
                              case  a if a==i => a
                              case _ =>  0
                            }
              val m = new NonEmptyMat(res.map(_.toVector).toVector)
              val updated = acc.updateMaxWithDelta(Tuple2(i, error(downSampledX, downSampledY)), r.get)
             r.get match {
               case r2 @ Rectangle(x,y,this.sWidth, this.sHeigh) =>
                 propagateError(None, updated , 0, 0)
               
               case r3 @ Rectangle(x,y,this.sWidth, h) =>
                 propagateError(r3.lineDownNonOverlap(), updated , 0, downSampledY+1)
                 
               case r1@ _ /*r1 @ Rectangle(0,y,w,h)*/ => 
                 propagateError(r1.goRightNonOverlap, updated, downSampledX+1, downSampledY)
               
             }
      }
      
      (downSampled, error) match {
        case (down,err) if (down.sHeigh == err.sHeigh && down.sWidth == err.sWidth) =>  propagateError(Option(Rectangle(0,0,windowX, windowY)), s ,0 , 0)
        case _ => throw PoolingException(POOLING_UPSAMPLE_MAT_SIZE)
      }
      
    }
    
    
    
  def updateMaxWithDelta(max : Tuple2[Double, Double], rect : Rectangle) : NonEmptyMat = {
    val res = for(x <- 0 to s.width-1)
      yield for(y <- 0 to s.heigh-1)
            yield (x,y) match {
               case a if a._1.isBetween(rect.x, rect.width) 
                      && a._2.isBetween(rect.y, rect.heigh) =>
                      
                        if(s(a._1, a._2) == max._1) max._2 else  0
               case b@ _ => s(b._1, b._2)
              }
   new NonEmptyMat(res.map(_.toVector).toVector)
  }
    


    
    
    @tailrec
    private def compute(rect : Option[Rectangle], acc : NonEmptyMat = new NonEmptyMat): NonEmptyMat = rect  match {
      
      case None => acc
      case _ => 
          
          val cuttedX = s.get.map( t => t.slice(rect.get.x, rect.get.width))
          val cuttedXY = cuttedX.slice(rect.get.y , rect.get.heigh)
          val max = cuttedXY.flatten.max
          
          rect match{
            
            case Some(Rectangle(x, y, this.sWidth, this.sHeigh)) => val appended = if(acc.heigh==0) Mat.wrapNumeric(max)
                                                                    else acc.apElemAt(max, acc.heigh-1)
                                                                    compute(None, appended)
                                                                    
            case Some(Rectangle(x, y, this.sWidth, h)) =>  val appended = if(acc.heigh==0) Mat.wrapNumeric(max)
                                                           else acc.apElemAt(max, acc.heigh-1)
                                                           compute(rect.get.lineDownNonOverlap(), appended)
                                                                    
            case Some(Rectangle(0, y, w , h)) =>  val newAcc = acc.ap(Vector(max))
                                                  compute(rect.get.goRightNonOverlap , newAcc)
                                                  
            case _ =>  val appended = acc.->(acc.heigh-1):+ (max)
                       compute(rect.get.goRightNonOverlap() , acc.updated(acc.heigh-1, appended))
            }
      }
    }
}
    
  
