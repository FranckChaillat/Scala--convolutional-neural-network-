package cnn.core.structure

import scala.annotation.tailrec
import scala.collection.mutable.WrappedArray
import cnn.helper.implicits.Implicits._
import cnn.core.objects._
import cnn.exceptions.{InvalidMatSizeException, MatTypeException}
import cnn.helper.implicits.Implicits.RichList
import cnn.activation.functions.{ActivationFunction, _SIGMOID, _SOFTMAX}
import cnn.exceptions.InvalidActivationFuncException
import cnn.exceptions.{KERNEL_INVALID_ACTIVATION, MAT_ADAPT}
import cnn.exceptions.MatOperationException
import cnn.exceptions.MAT_OPERATION_FAILED
import cnn.exceptions.MatTypeException


abstract class Mat extends NeuralUnit

case class EmptyMat() extends Mat{
 
  val heigh = 0
  val width = 0
  
  def ->(index : Int) = throw new IndexOutOfBoundsException("the requested index ('"+index+"') doesn't exist into the Matrix")
  
  def %(x : Int, y : Int) = (width % x == 0 && heigh % y == 0)
  
  def ap(a: Vector[Double]) =   new NonEmptyMat(Vector(a))
  
  def pr(a : Vector[Double]) =  new NonEmptyMat(Vector(a))
}



 class NonEmptyMat(val s : Vector[Vector[Double]]) extends Mat{
 
  val heigh = s.size
  val width = if(s.size> 0) s(0).size else 0
  
  val indexW = if (width!= 0) width -1 else 0
  val indexH = heigh-1
   
  def this() = this(Vector())
  
  def ->(index : Int) = if(index <= s.size) 
                               s(index)
                        else throw new IndexOutOfBoundsException("the requested index ('"+index+"') doesn't exist into the Matrix")
  
  def apply(x: Int, y : Int) =  s(x).toSeq(y)
  
  def %(x : Int, y : Int) =  (width % x == 0 && heigh % y == 0)
  
  def get = s
  
  def ap(a: Vector[Double]) =  new NonEmptyMat(s.:+(a))
  
  def pr(a : Vector[Double]) = new NonEmptyMat(s.+: (a))
  
  def updated(i : Int, e : Vector[Double]) = new NonEmptyMat(s.updated(i, e))
  
  def apElemAt(e : Double, i : Int) = {  
    val res = s(i) ++ Vector(e)
    val up = s.updated(i, res)
     new NonEmptyMat(up)
  }
  
  
  def multiply = combineMat((x,y) => x*y)
  def add = combineMat((x,y) => x+y)
  def divide = combineMat((x,y) => x/y)
  def substract = combineMat((x,y) => x-y)
    
  private def combineMat(op : (Double, Double) => Double) : (Vector[NonEmptyMat]) => NonEmptyMat =
  {
    @tailrec
    def combineF(acc : NonEmptyMat)(lst : Vector[NonEmptyMat]) : NonEmptyMat = lst.headOption match{
      case None => acc
      case Some(mat) if acc.compareSize(mat) => 
        val res = mat.get.zip(acc.get).map{case (a,b) =>  Range(0,a.length).map(i=> op(a(i), b(i))).toVector}
        combineF(new NonEmptyMat(res)) (lst.tail)

      case _ => throw MatTypeException(MAT_OPERATION_FAILED)
    
    }
    combineF(this)
  }

  
  def getSubMat(rect : Rectangle) = rect match{
    case Rectangle(x , y, w , h) if w > this.width | h > this.heigh => 
      throw new IllegalArgumentException("The given rect object is bigger than the Mat")
      
    case _ => val cuttedX = s.map( t => t.slice(rect.x, rect.width))
              new NonEmptyMat(cuttedX.view(rect.y , rect.heigh).toVector)
  }
  
  def compare(m : NonEmptyMat) = {
    if(m.width != this.width && m.heigh != this.heigh)
      false
    else
       m.get.zip(this.get).forall(x=> x._1.sameElements(x._2))
  }
  
  def compareSize(m : NonEmptyMat) = (m.heigh == this.heigh && m.width == m.width)
  
  def compareSize(h: Int, w : Int) = (heigh==h && width==w)
  
  def rot180 = new NonEmptyMat(this.get.reverse.map(x=> x.reverse))
  
  def pad(value : Double, width : Int) = {
    
    val rows = (1 to width).map(x=> 
                          (1 to s.length +(2*width)).toVector.map(y => value)).toVector
    
    val res = s.map { x => 
      val cols = (for(i<- (1 to width)) yield value).toVector
      cols ++ (x ++ cols)
    }
    new NonEmptyMat(rows ++ (res ++ rows))
     
    
  }
   
}

case class Kernel(override val s : Vector[Vector[Double]], activationFunc : ActivationFunction = _SIGMOID, preActivation : Vector[NonEmptyMat] = Vector(), activation : Vector[NonEmptyMat] = Vector(), delta : Vector[NonEmptyMat] = Vector()) 
extends NonEmptyMat(s)
{
  
  def apply() = {
    def compute(m : NonEmptyMat, f : Double => Double) = new NonEmptyMat (m.get.map(x => x.map(x=> f(x))))
    
    activationFunc match {
      case sig @ `_SIGMOID` =>  updateWithActivation(preActivation.map(x=> compute(x,  (d : Double) => ( sig(d)) )))
      case `_SOFTMAX`=> throw InvalidActivationFuncException(KERNEL_INVALID_ACTIVATION)
    }
  }
  
  def computeDelta(m : Vector[NonEmptyMat]) = {
     def compute(m : NonEmptyMat, f : Double => Double) = new NonEmptyMat (m.get.map(x => x.map(x=> f(x))))
     
     activationFunc match {
       case sig @ `_SIGMOID` => m.map( x=> compute(x, (d : Double) => (sig.derivative(d))) )
       case `_SOFTMAX`=> throw InvalidActivationFuncException(KERNEL_INVALID_ACTIVATION)
       
     }
  }
  
  def update(v : Double) = Kernel(s.map(_.map (y => y+v)), activationFunc, preActivation, activation, delta )
  
  def updateWithPreActivation(preact : Vector[NonEmptyMat]) = Kernel(s, activationFunc, preact, activation, delta)
  
  def updateWithActivation(act : Vector[NonEmptyMat]) = Kernel(s, activationFunc, preActivation, act, delta)
  
  def updateWithDelta(_delta : Vector[NonEmptyMat]) = Kernel(s,activationFunc, preActivation, activation, _delta)
  
  override def rot180 = new Kernel(this.get.reverse.map(x=> x.reverse),activationFunc, preActivation, activation, delta)
}


object Kernel{
  def wrapNumeric(n : Double) = Kernel(Vector(Vector(n)))
  def fromMatrix(m : NonEmptyMat) = Kernel(m.s)
}


object Mat{
    
    def apply(n : Double) = wrapNumeric(n)
    
    def wrapNumeric(n : Double) = new NonEmptyMat(Vector(Vector(n)))
    
    def fillMatWith(value : Double, width: Int, heigh : Int) = new NonEmptyMat(Range(0,heigh).map(x => Range(0, width).map(x=>value).toVector).toVector)
    
    def reduce(v : Vector[Double]*) = v.map(_.foldLeft(0.0)((acc, e)=> acc+e)).sum
    
    
    
                    
     
    def toNeurons(m: NonEmptyMat) = 
        m.get.flatMap(_.map { x => new Neuron(Vector(), Vector(), _SIGMOID, x) })
        
    
    def completeMatWith(m : Mat, heigh : Int, width : Int, value : Double) =  m match {
        case e : EmptyMat => Mat.fillMatWith(value, width, heigh)
        case x : NonEmptyMat  => 
                if(x.heigh <= heigh && x.width <= width){
                       val res = x.get.map(e =>  e ++ ( Vector.fill(width - e.size)(value) ))
                       Range(0, heigh-x.heigh).foldLeft(new NonEmptyMat(res)) ((acc, e) => acc.ap(Vector.fill(width)(value)))
                } else x    
      }
    
    
}

