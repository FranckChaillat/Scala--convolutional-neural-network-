package cnn.core.structure

import cnn.exceptions.NeuralLinkException
import cnn.exceptions.{LINK_CONSIST_LENGTH, LINK_CONSIT, LINK_WEIGHT_MULT}
import cnn.exceptions.NeuralLinkException
import scala.annotation.tailrec

case class Link(input: Double, out: Option[Neuron], weight : Double){
  
  def this(w : Double) = this(0, None, w)
  def this(input : Double, weight : Double) = this(input, None, weight)

  def updateWithInput(i : Double) = Link(i, out , weight)
  
  def updateWeight(learningRate : Double, n : Neuron) = {
    Link(input, out, weight - (learningRate * (-n._der * input)))
  }
  
  def * =  input * weight
  def rev = out match {
    case None => throw new NeuralLinkException(LINK_CONSIT)
    case Some(x) => x._der * weight
  }
}
