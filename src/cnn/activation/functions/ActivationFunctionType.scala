package cnn.activation.functions

import cnn.core.structure.OutNeuron


sealed trait ActivationFunction { def value : String }

case object _SOFTMAX extends ActivationFunction {
  def value = "SOFTMAX"
  def apply(parentLayerPreact : Seq[Double], mainPreact : Double) = {
     val total = parentLayerPreact.foldLeft(0.0)((Acc, x)=> Acc +  scala.math.exp(x))
     scala.math.exp(mainPreact)/total
  }
      
  def derivative(n : OutNeuron, target : Int) = n match {
    case x : OutNeuron if x.classification==target => 1.0 - x.act
    case _ => 0.0 - n.act
  }
}


case object _SIGMOID extends ActivationFunction {
  
  def value = "SIGMOID"
  def apply(preactivation : Double) = 1/(1+ scala.math.exp(-preactivation))
  def derivative(activation : Double) = activation * (1-activation)  

}

