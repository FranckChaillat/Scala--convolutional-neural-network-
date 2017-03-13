package cnn.core.structure

import cnn.core.convolution._
import cnn.core.convolution.Convolution._
import cnn.core.subsample.SubSampling._
import cnn.exceptions.{MatCountException, PoolingException, MatTypeException, ConvolutionDerivativeException, FCLayerStructureException, NeuralLinkException}
import cnn.core.subsample.SubSamplingMethod
import scala.annotation.tailrec
import cnn.helper.implicits.Implicits.RichList
import scala.Vector
import cnn.exceptions.LayerTypeException
import cnn.learning.LearningContext
import cnn.exceptions.InvalidNeuralUnitTypeException
import cnn.learning.LearningContext
import cnn.exceptions.{CONV_KERNEL_COUNT, CONV_DIV, CONV_NEXTLAYER_MISS, KERNEL_UPDATE_FAIL, KERNEL_INVALID_ACTIVATION,
                      INVALID_LAST_LAYER, INVALID_LAYER_ORDER, POOLING_SUM, UPSAMPLING_MAT_COUNT, FC_NEURON_CONSIST, FC_LINK_COUNT, NOT_FC, POOLING_CONST, LINK_CONSIT}
import cnn.learning.LearningContext
import cnn.exceptions.MatTypeException
import cnn.exceptions.LayerTypeException



/***Base class***/
 class Layer[+A <: NeuralUnit](units : Vector[A]= Vector()) {
  
  def lenght = units.length
  def head = units.head
  def tail = units.tail.toSeq
  def -> (i : Int =0) = units(i)
  def :+(u : NeuralUnit) = units.:+(u)
  def +:(u : NeuralUnit) = units.+:(u)
  def get = units
  def getRange(from : Int, to : Int) = if(from>= to) 
                                          throw new IllegalArgumentException("the 'to' parameter needs to be greater than the 'from'")
                                       else units.take(to).takeRight(from)
                           
}


/***Case classes***/
case class InputLayer(in : Vector[NonEmptyMat]) extends Layer[NonEmptyMat](in)

case class ConvolutionLayer(kernel : Vector[Kernel]) extends Layer[NonEmptyMat](kernel) with ProcessableLayer[NonEmptyMat] {
  
  override def get = kernel
  
  def apply(input : Layer[NonEmptyMat]) = {
    
      val res = input match {
       case in : InputLayer if(input.lenght == kernel.length) => 
                                  kernel.par.zip(input.get)
                                        .map(x=> x._1.updateWithPreActivation(Vector(x._2*(x._1, _VALID))))
                                        .map(x=> x())

       case _ =>  kernel.par.map(x => x.updateWithPreActivation(input.get.map(y => y.*(x, _VALID))))
                            .map(p=> p())
      }
      ConvolutionLayer(res.seq)
  }

  
  def derivate[B<: NeuralUnit](nextLayer : Option[ProcessableLayer[B]], lc : LearningContext) =
    nextLayer.fold(throw ConvolutionDerivativeException(CONV_NEXTLAYER_MISS))( x=>x match {
       case c : ConvolutionLayer =>  
           val nextLayerDelta = getNextLayerDelta(x).flatMap(_.get)
           val res = kernel.par.map { k => 
                         val rot = k.rot180
                         val conv = nextLayerDelta.map { x => x.*(rot, _FULL) }
                         rot.computeDelta(conv)
                     }
           val updated = for(i <- 0 to kernel.size -1)
                             yield kernel(i).updateWithDelta(res(i))
           ConvolutionLayer(updated toVector)                             
             
       case l @ _ =>
           val nextLayerDelta = getNextLayerDelta(x).flatMap(_.get)
           val res = nextLayerDelta.grouped(nextLayerDelta.size/kernel.size).toVector.par
                                   .zip(kernel.map(_.rot180))
                                   .map(x=> x._2.computeDelta(x._1.map(y=> y *(x._2, _FULL))))
           val updated = for(i<- 0 to kernel.size-1)
                           yield kernel(i).updateWithDelta(res(i))
           ConvolutionLayer(updated toVector)
       })
    
       
 
   def updateKernel[T <: NeuralUnit](nextLayer : ProcessableLayer[T], lc : LearningContext) = { 
        val nextLayerDelta = getNextLayerDelta(nextLayer).flatMap(_.get)
        val kernelActivation = kernel.flatMap(x => x.activation)
        val multiplied =  nextLayerDelta.zip( kernelActivation).par
        val res = multiplied.map{ case (a,b) => a.multiply(Vector(b))}
                            .map(x=> x.*(Kernel.wrapNumeric(lc.leaningRate), _VALID))
                            .seq.grouped(nextLayerDelta.size / kernel.size).toVector
                            .flatMap(_.map(x=> Mat.reduce(x.get : _* )))
         
        ConvolutionLayer((for(i<- 0 to kernel.size-1)
                            yield kernel(i) update(res(i))).toVector)
  }
       

  
  def getDelta = kernel.map(x => Layer(x.delta))
  
  def getActivation = Some(Layer(kernel.flatMap(x => x.activation))) //TODO : check if Option type is necessary
  

}
  
case class PoolingLayer(window : PoolingWindow , method : SubSamplingMethod) extends Layer[NonEmptyMat] with ProcessableLayer[NonEmptyMat]{
  
    def apply(input : Layer[NonEmptyMat]) : PoolingLayer = {
      val res = input.get map (x => x.max(Tuple2(window.x, window.y)))
      val newWindow = new PoolingWindow(window.x, window.y, input, new Layer(res))
      PoolingLayer(newWindow, method)
    }
    
    
    /**Get the PoolingWindow object and compute the error matrix for each matrix in it**/
    def derivate[B<: NeuralUnit] (nextLayer : Option[ProcessableLayer[B]], lc : LearningContext) : PoolingLayer = window match {
          case PoolingWindow(x,y,i,a) => nextLayer match {
            case None    => throw new PoolingException(INVALID_LAST_LAYER)
            case Some(e) =>

                val nextLayerDelta = getNextLayerDelta(e).par
                val res = nextLayerDelta.map(e=> upsample(x,y, i.get ,a.get, e.get))
                val sum = RichList.powerZip[NonEmptyMat](res.seq)
                                    .map{case h+: t => h.add(t)}
                                    //.map(x=>  Mat.combineMat(x, Mat._MATADD))
                                  .collect{ case x : NonEmptyMat => x
                                            case _ => throw PoolingException(POOLING_SUM)
                                          }
                updateWindow(new PoolingWindow(x,y,i,a, Layer.apply(sum)))
            
          }
       }
      
    
    
    @tailrec
    private def upsample(x : Int, y : Int, input : Vector[NonEmptyMat] ,base : Vector[NonEmptyMat], err : Vector[NonEmptyMat], acc : Vector[NonEmptyMat] = Vector()) : Vector[NonEmptyMat] = PoolingLayer.zip(input, base, err) match {
      case h +: t  if base.length == err.length => 
        upsample(x, y, input.tail, base.tail, err.tail, acc.:+( h._1.upsample(x, y, h._2,  h._3)))
      
      case Nil  =>  acc
      case _ => throw PoolingException(UPSAMPLING_MAT_COUNT)
    }
    
    def getActivation() : Option[Layer[NonEmptyMat]] = if(window.activation.lenght>0) Some(window.activation) else None
    
    def getDelta() : Vector[Layer[NonEmptyMat]] = Vector(Layer(window.delta.get))
    
    def updateWindow(pw : PoolingWindow) : PoolingLayer = new PoolingLayer(pw, method)
}
  
  
case class FCLayer(neurons : Vector[Neuron]) extends Layer[Neuron](neurons) with ProcessableLayer[Neuron]{
  
  neurons match {
     case h +: t if !t.forall(x => x._activationFun == h._activationFun)&&
                    !t.forall(x =>  x.getClass.isInstance(h.getClass))=> 
         throw FCLayerStructureException(FC_NEURON_CONSIST)
     case Vector() => throw FCLayerStructureException(FC_NEURON_CONSIST)
     case _ => neurons
   
  }
  
  override def get = neurons
  
  def getActivation(): Option[Layer[NonEmptyMat]] = if(neurons.size >0) Some(Layer(neurons.map(x=> Mat.wrapNumeric(x._act)))) else None
  
    def apply(input : Layer[Neuron]) : FCLayer = neurons match {
       case a@ h +:t =>
          val updated = a.map( n => if(n._inLinks.length == input.get.length)
                                      n.updateWithInput( input.get.map(_._act).zip(n._inLinks)
                                                                  .map(x=> x._2.updateWithInput(x._1)))
                                    else throw NeuralLinkException(FC_LINK_COUNT))
          updated.head match {
               case x: OutNeuron => val preacts  = updated.collect({
                                              case x: OutNeuron => x.updateWithPreact(x.computePreactivation)
                                              case _ => throw new FCLayerStructureException(FC_NEURON_CONSIST)
                                            })
                                     FCLayer(preacts.map(x => x(Layer(preacts))))
               case _ => FCLayer(a.map(_()))
         }
                                 
     
    case _ => throw FCLayerStructureException(FC_NEURON_CONSIST)
  }
  
  def derivate[B<: NeuralUnit](nextLayer : Option[ProcessableLayer[B]], lc : LearningContext) = 
    nextLayer.fold(FCLayer( neurons.collect{
                      case x: OutNeuron => x.updateWithDerivative(x.computeDelta(lc.target))
                      case _ => throw InvalidNeuralUnitTypeException(NOT_FC)
                     })
                  )(x => x match {
                      case fc : FCLayer => val der = neurons.map(x=> x.updateWithDerivative(x.outLinks.foldLeft(0.0)((i,j)=> i + j.rev)))
                                           FCLayer(der)
                      case _ => throw LayerTypeException(INVALID_LAYER_ORDER)  
                   })
  
  def updateWeight(lc : LearningContext) =  
     FCLayer(neurons.map(n => { 
               val lnk = n._inLinks.map(l => l.updateWeight(lc.leaningRate, n))
               n.updateWithInput(lnk)
             }))
  
  
  def getRawDelta = {
          val res = (0 to neurons.head._inLinks.size-1).map(l => {
  
                                           val reduced = (0 to neurons.size-1).foldLeft(0.0)((Acc,n) => {
                                             Acc + (neurons(n)._inLinks(l).weight * neurons(n)._der)
                                           })
                                        
                     Mat.wrapNumeric(reduced)
                    }).toVector
          Vector(Layer(res))
  }
             
  def getDelta = Vector(Layer (neurons.map { x => Mat.wrapNumeric(x._der)} ))           
             
  
}
 
 
 /***Companion***/
 
 object Layer{
   def getEmpty[B <: NeuralUnit]() : Layer[B] = new Layer()
   def apply[B <: NeuralUnit](mat : Vector[B]) = new Layer(mat)

 }
 
 object InputLayer{
   def getLayered(mat : Vector[NonEmptyMat]) : InputLayer = InputLayer(mat)
 }
 
 object ConvolutionLayer{
   def getLayered(mat : Vector[Kernel]) : Layer[NonEmptyMat] = ConvolutionLayer(mat)
 }
 
 object PoolingLayer{
   def zip(input : Vector[NonEmptyMat], activation : Vector[NonEmptyMat], error : Vector[NonEmptyMat]): List[Tuple3[NonEmptyMat, NonEmptyMat, NonEmptyMat]] = {
    if(input.length == activation.length && activation.length == error.length){
     val res = for(i <- 0  to input.length-1)
                 yield (input(i), activation(i), error(i))
     res.toList
    }else throw PoolingException(POOLING_CONST)
     
   }
 }
 