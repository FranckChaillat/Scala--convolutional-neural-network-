package cnn.core.structure

import cnn.learning.LearningContext
import cnn.exceptions.{BackPropagationException, ActivationException}
import cnn.exceptions.{BP_ERROR, NO_ACTIVATION}
import cnn.exceptions.MatCountException

trait ProcessableLayer[A <: NeuralUnit] {
  def derivate[B <: NeuralUnit](nextLayer : Option[ProcessableLayer[B]], lc : LearningContext) : Layer[A]
  def apply(input : Layer[A]) : Layer[A]
  def getDelta() : Vector[Layer[NonEmptyMat]]
  def getActivation() : Option[Layer[NonEmptyMat]]
  
  def getNextLayerDelta[B <: NeuralUnit](nextLayer : ProcessableLayer[B]) =  nextLayer match {
    case  fc : FCLayer => 
          val act = this.getActivation
          val delta = fc.getDelta
          if(delta.size == 0 || act.get.lenght ==0)  throw BackPropagationException(BP_ERROR)
          else {
               val delta = fc.getDelta
                 
                               val t = delta.head.get.map { x => x(0,0) }
                               val t2 = t.grouped(t.size / act.get.lenght).toVector
                                         .map(_.grouped(act.get.head.width).toVector)
                                         .map(x=> new NonEmptyMat(x)).toVector
              Vector(Layer(t2))
           }
                                 
    case _ => nextLayer.getDelta
  }


}