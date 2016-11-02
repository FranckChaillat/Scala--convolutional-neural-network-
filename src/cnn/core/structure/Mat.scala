package cnn.core.structure

import scala.annotation.tailrec
import scala.collection.mutable.WrappedArray
import cnn.helper.implicits.Implicits._
import cnn.core.objects._
import cnn.exceptions.InvalidMatSizeException
import cnn.exceptions.MatTypeException
import cnn.exceptions.InvalidMatSizeException
import cnn.activation.functions._SIGMOID
import cnn.helper.implicits.Implicits.RichList
import cnn.activation.functions.ActivationFunction
import cnn.activation.functions.Sigmoid
import cnn.activation.functions._SOFTMAX
import cnn.exceptions.InvalidActivationFuncException
import cnn.exceptions.{KERNEL_INVALID_ACTIVATION, MAT_ADAPT}



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
  
  def rot180 = new NonEmptyMat(this.get.reverse.map(x=> x.reverse))
   
}

case class Kernel(override val s : Vector[Vector[Double]], activationFunc : ActivationFunction = _SIGMOID, preActivation : Vector[NonEmptyMat] = Vector(), activation : Vector[NonEmptyMat] = Vector(), delta : Vector[NonEmptyMat] = Vector()) 
extends NonEmptyMat(s)
{
  
  def apply() = {
    def compute(m : NonEmptyMat, f : Double => Double) = new NonEmptyMat (m.get.map(x => x.map(x=> f(x))))
    
    activationFunc match {
      case `_SIGMOID` =>  updateWithActivation(preActivation.map(x=> compute(x,  (d : Double) => ( Sigmoid(d)) )))
      case `_SOFTMAX`=> throw InvalidActivationFuncException(KERNEL_INVALID_ACTIVATION)
    }
  }
  
  def computeDelta(m : Vector[NonEmptyMat]) ={
     def compute(m : NonEmptyMat, f : Double => Double) : NonEmptyMat = new NonEmptyMat (m.get.map(x => x.map(x=> f(x))))
     activationFunc match {
       case `_SIGMOID` => m.map( x=> compute(x, (d : Double) => (Sigmoid.derivative(d))) )
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
}


object Mat{
    
    def apply(n : Double) = wrapNumeric(n)
    
    def wrapNumeric(n : Double) = new NonEmptyMat(Vector(Vector(n)))
    
    def fillMatWith(value : Double, width: Int, heigh : Int) = new NonEmptyMat(Range(0,heigh).map(x => Range(0, width).map(x=>value).toVector).toVector)
    
    def reduce(v : Vector[Double]*) = v.map(_.foldLeft(0.0)((acc, e)=> acc+e)).sum //TODO : Tester reduce
    
    
    
    /**Operations available on Matrices**/
    sealed class MatOperation(val operation : (Double, Double ) => Double)
    case object _MATDIV extends MatOperation((x,y) => x/y)
    case object _MATMUL extends MatOperation((x,y) => x*y)
    case object _MATADD extends MatOperation((x,y) => x+y)
    case object _MATSUB extends MatOperation((x,y) => x-y)
    
    /**Combine all matrices using the specified math operation**/
    def combineMat(lst : Vector[NonEmptyMat], op : MatOperation): Mat = {
      
      @tailrec
      def compute(l : Vector[NonEmptyMat], acc : NonEmptyMat) : NonEmptyMat = l.headOption match{
        case None => acc
        case Some(mat) => 
          val add = mat.get.zip(acc.get).collect{
            case (a,b) if a.length == b.length => Range(0,a.length).map(i=> op.operation(a(i), b(i))).toVector
            case _ => Vector()
          }
          compute(l.tail, new NonEmptyMat(add))
        }
    
      
      lst.headOption match {
        case Some(mat) if lst.forall(x=> x.compareSize(mat)) => op match {
          case `_MATMUL` |`_MATDIV` => compute(lst, Mat.fillMatWith(1, mat.width, mat.heigh))
          case _ => compute(lst, Mat.fillMatWith(0, mat.width, mat.heigh))
        } 
        case None => new EmptyMat()
      }
    }                    
      
    //TODO : Test
    def toNeurons(m: NonEmptyMat) = 
        m.get.flatMap(_.map { x => new Neuron(Vector(), Vector(), _SIGMOID, x) })
    
    
}

