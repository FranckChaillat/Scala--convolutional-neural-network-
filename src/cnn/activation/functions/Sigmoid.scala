package cnn.activation.functions

import cnn.core.structure.NeuralUnit
import cnn.core.structure.NonEmptyMat
import cnn.core.structure.Neuron

object Sigmoid {
  
    def apply(preactivation : Double) : Double = 1/(1+ scala.math.exp(-preactivation))
    def derivative(activation : Double) = activation * (1-activation)

}
