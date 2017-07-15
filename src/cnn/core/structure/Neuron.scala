package cnn.core.structure


import cnn.activation.functions._
import cnn.exceptions.InvalidActivationFuncException
import scala.annotation.tailrec
import cnn.exceptions.NeuralLinkException
import cnn.exceptions.MatCountException
import cnn.exceptions.FCLayerStructureException
import cnn.exceptions.NeuronActivationException
import cnn.exceptions.{FC_LINK_COUNT, NEURON_INVALID_ACTIVATION}




class Neuron(inLinks : Vector[Link], val outLinks : Vector[Link],  activationFun : ActivationFunction, act : Double = 0, preact : Double =0 ,  derivative : Double =0) 
  extends NeuralUnit
{
  
  val _inLinks = inLinks
  val _activationFun = activationFun
  val _act = act
  val _der = derivative
  val _preact = preact
  
  def this(inLinks : Vector[Link], activationFun : ActivationFunction) =
      this(inLinks, Vector(), activationFun, 0,0,0)
  
  
  def apply() = inLinks match{
      case h +: t => val act = computeActivation(computePreactivation())
                     updateWithActivation(act)
      case _      => throw NeuralLinkException(FC_LINK_COUNT)
  }
  
  //changed
  private def computeActivation(preactivation : Double) = this.activationFun match {
      case `_SOFTMAX` =>  throw InvalidActivationFuncException(NEURON_INVALID_ACTIVATION)
      case sig @ `_SIGMOID` => sig(preactivation) 
    }
  
  
  def computePreactivation() = inLinks.foldLeft(0.0)((x,y)=> x + y.*)
  
  def computeDelta() = _activationFun match {
      case `_SOFTMAX` =>  throw InvalidActivationFuncException(NEURON_INVALID_ACTIVATION)
      case sig @ `_SIGMOID` => sig.derivative(_act)
  }

  
  def updateWithDerivative(d : Double) = new Neuron(inLinks, outLinks, activationFun, act, preact, d)
  
  def updateWithActivation(a : Double) = new Neuron(inLinks, outLinks, activationFun, a, preact, derivative)
  
  def updateWithInput(in : Vector[Link]) = new Neuron(in, outLinks, activationFun, act, preact, derivative)
}


/**Output layer neuron**/

case class OutNeuron(inLinks : Vector[Link], activationFun : ActivationFunction, classification : Int, act : Double , preact : Double , derivative : Double) 
  extends Neuron(inLinks, Vector(), activationFun, act, preact, derivative){
  
  def this(_inLinks : Vector[Link], _classification: Int) = this(_inLinks, _SOFTMAX, _classification,0,0,0)
  
  def apply(layerPreact : Layer[OutNeuron]) = inLinks match {
    case h+:t => val act = computeActivation(layerPreact.get.map(_.preact))
                 updateWithActivation(act)
    case _    => throw new NeuralLinkException(FC_LINK_COUNT)
  }

  private def computeActivation( layerPreact : Seq[Double])  = this.activationFun match {
      case soft @ `_SOFTMAX` => soft(layerPreact, preact)
      case sig @ `_SIGMOID` =>  sig(preact)
    }
  
 
  def computeDelta(target : Int) = activationFun match{
      case soft @ `_SOFTMAX` => soft.derivative(this, target)
      case sig @ `_SIGMOID` => sig.derivative(act)
  }
  
  override def updateWithActivation(a : Double) = OutNeuron(inLinks, activationFun, classification, a, preact, derivative)
  override def updateWithDerivative(d: Double) = OutNeuron(inLinks, activationFun ,classification, act, preact, d)
  override def updateWithInput(in : Vector[Link]) =  OutNeuron(in, activationFun, classification ,act, preact, derivative)
  def updateWithPreact(p : Double) = OutNeuron(inLinks, activationFun, classification, act, p, derivative)
  
}

/**Companion**/
object Neuron{
  def unapply(n : Neuron) : Option[ActivationFunction] = Some(n._activationFun)
}