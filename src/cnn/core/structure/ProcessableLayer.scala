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
          if(fc.getDelta.size == 0 || act.get.lenght ==0)  throw BackPropagationException(BP_ERROR)
          else {
                 val delta = fc.getRawDelta
                 val values = delta.head.get.map { x => x(0,0) }
                 val res = values.grouped(values.size / act.get.lenght).toVector
                           .map(_.grouped(act.get.head.width).toVector)
                           .map(x=> new NonEmptyMat(x)).toVector
              Vector(Layer(res))
           }
                                 
    case _ => nextLayer.getDelta
  }


}