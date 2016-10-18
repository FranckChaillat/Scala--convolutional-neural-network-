package cnn.learning.serialization

import cnn.core.structure._

object Writer {

  
  def write(path : String, net : Network) = {
    
   def record(acc : String, l : Seq[Layer[NeuralUnit]]) = l match {
     case Seq() => acc
     case x :+ xs => x match {
                       case conv : ConvolutionLayer =>  conv.kernel
                     }
   }
  }
  

  
}