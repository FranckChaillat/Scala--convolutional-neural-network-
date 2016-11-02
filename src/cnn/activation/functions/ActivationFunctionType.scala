package cnn.activation.functions

sealed trait ActivationFunction { def value : String}
case object _SOFTMAX extends ActivationFunction { val value = "SOFTMAX"}
case object _SIGMOID extends ActivationFunction { val value = "SIGMOID"}

