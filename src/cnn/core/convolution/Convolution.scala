package cnn.core.convolution
import scala.annotation.tailrec
import cnn.helper.implicits.Implicits._
import cnn.core.structure.{NonEmptyMat, Mat, EmptyMat}
import cnn.core.objects.Rectangle
import cnn.exceptions.InvalidMatSizeException
import cnn.helper.implicits.Implicits._
import cnn.exceptions.ConvolutionException
import cnn.core.structure.{EmptyMat,NonEmptyMat,Layer}
import cnn.exceptions.{CONV_KERNEL_COUNT,CONV_DIV,CONV_INVALID_RES, CONV_KERNEL_FORMAT_ERR}
import cnn.core.objects.Rectangle
import cnn.exceptions.ConvolutionException




object Convolution{
  
  case class AccumulatorParam(base : Mat, rect : Rectangle) {
    def add(a: Double, r : Rectangle) : AccumulatorParam = base match {
      
      case y : EmptyMat => AccumulatorParam( new NonEmptyMat(Vector(Vector(a))), rect)
      case x : NonEmptyMat if(r.y > rect.y) =>  AccumulatorParam(x.ap(Vector(a)), r)
      case y : NonEmptyMat =>  val last = base.->(y.heigh-1)
                               val appended = y.->(y.heigh-1):+(a)
                               AccumulatorParam( y.updated(y.heigh-1, appended ), r)
    }
    
  }
 
  
  implicit class Matrix(s : NonEmptyMat) extends Rectangle(0,0, s.width, s.heigh){
    
    private val sSeq = s.get
    
    def split(kernelCnt : Int) : Vector[NonEmptyMat] = kernelCnt match {
      case a if a.isPrime => throw ConvolutionException(CONV_KERNEL_COUNT)
      case _ =>  {
        val sets =Range(1,s.width+1 ) .flatMap(x => {
                   Range(1, s.heigh+1 ) filter(y => (x * y == kernelCnt) && s.width % x==0 && s.heigh % y==0) map( e => (x,e))
      })
                
        val div = sets match {
                    case lst @ x+:xs => 
                      lst minBy(_ match {
                       case (a,b) if (a>=b) => a - b
                       case (a,b) if (a<b) => b - a
                      })
                    case _ => throw ConvolutionException(CONV_DIV)
                }

         divide(Vector(), Option(Rectangle(0,0, s.width/ div._1, s.heigh/ div._2)))
      }
    }
    

    @tailrec
    private def divide(acc : Vector[NonEmptyMat] , rect : Option[Rectangle]) : Vector[NonEmptyMat] = rect match {
      case Some(Rectangle(x, y, this.width,h)) => h match {
        case this.heigh => acc.:+ (s.getSubMat(rect.get))
        case _ => divide( acc.:+(s.getSubMat(rect.get)) , rect.get.lineDownNonOverlap)
      }
      
      case _ => divide( acc.:+(s.getSubMat(rect.get)) , rect.get.goRightNonOverlap)
      
    }
    
    
    
    def *(kernel : NonEmptyMat, method : ConvolutionMethod) =
      
      if(kernel.width > s.width || kernel.heigh > s.heigh)
        throw ConvolutionException(CONV_KERNEL_FORMAT_ERR)
      else
        
        method match {
        case  `_VALID` => validConv(kernel , Option.apply(Rectangle(0,0,kernel.width, kernel.heigh)),
                                      AccumulatorParam(EmptyMat(), Rectangle(0,0, kernel.width, kernel.heigh)))
        case  `_FULL` =>  fullConv(kernel , Option.apply(Rectangle(1- kernel.width ,1- kernel.heigh, 1 ,  1)),
                                      AccumulatorParam(EmptyMat(), Rectangle(0,0, kernel.width, kernel.heigh)))                          
        }
    
    
    //TODO : optimize
    @tailrec
    private def validConv(kernel :  NonEmptyMat, rect : Option[Rectangle], result: AccumulatorParam) : NonEmptyMat = rect match {
        case None => result.base.asInstanceOf[NonEmptyMat]
        case _ => { 
              val cuttedX = sSeq.map( t => t.slice(rect.get.x, rect.get.width))
              val cuttedXY = cuttedX.slice(rect.get.y , rect.get.heigh).zipWithIndex
              val res = cuttedXY.map(x => compute(x._1, kernel ->(x._2)))
              
              rect match{
                case Some(Rectangle(x, y, this.width ,h)) => h match {
                  case this.heigh => validConv(kernel, None, result.add(res.sum, rect.get))
                  case _=> validConv(kernel, rect.get lineDown, result.add(res.sum, rect.get))
                }
                case _ =>  validConv(kernel, rect.get.goRight, result.add(res.sum, rect.get))
              }
        }
    }
   
    
   @tailrec
   private def fullConv(kernel : NonEmptyMat, rect : Option[Rectangle], result : AccumulatorParam) : NonEmptyMat = rect match {
     case None => result.base match {
       case x: NonEmptyMat => x
       case _ => throw ConvolutionException(CONV_INVALID_RES)
     }
     case Some(r) =>  {
              val cutted = sSeq.map(_.slice(rect.get.x, rect.get.width))
                               .slice(rect.get.y , rect.get.heigh).zipWithIndex
              val cuttedK = cutkernel(kernel, rect.get, this)
              val res = cutted.map(x => compute(x._1, cuttedK ->(x._2)))
            
       r match {
         case Rectangle(x,y,w,h) if w == (width + (rect.get.getRectWidth-1)) 
                                 && h == (heigh + (rect.get.getRectHeigh-1)) => fullConv(kernel, None, result.add(res.sum, rect.get))       
                
         case Rectangle(x, y, w, h) if w == (width + (rect.get.getRectWidth-1)) => fullConv(kernel, rect.get.lineDownFull, result.add(res.sum, rect.get))
           
         case  _ =>  fullConv(kernel, rect.get.goRight, result.add(res.sum, rect.get))
       }
     }
   } 
   
   
   private def cutkernel(k : NonEmptyMat, kRect: Rectangle, mRect : Rectangle) = {
     val cutX = if (kRect.x < mRect.x) 
                   k.get.map(_.drop( mRect.x - kRect.x ))
                else if(kRect.width > mRect.width) 
                   k.get.map(_.dropRight(kRect.heigh - mRect.heigh))
                else k get
     
     val cutxy = if(kRect.y < mRect.y)
                   cutX.drop(mRect.y - kRect.y)
                 else if(kRect.heigh > mRect.heigh)
                   cutX.dropRight(kRect.heigh - mRect.heigh)
                 else cutX
     new NonEmptyMat(cutxy)            
           
   }
     
     
     
     
   
   private def compute(m1 : Seq[Double], m2 : Seq[Double]) : Double =  m1.zip(m2).map((x)=> x._1 * x._2).sum
    
  }
}

