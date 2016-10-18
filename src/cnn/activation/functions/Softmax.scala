package cnn.activation.functions

import cnn.core.structure.OutNeuron


object Softmax {
  
    def apply(parentLayerPreact : Seq[Double], mainPreact : Double) : Double ={
      val total = parentLayerPreact.foldLeft(0.0)((Acc, x)=> Acc +  scala.math.exp(x))
      //val total = parentLayerPreact.reduce((x,y) => x + scala.math.exp(y))
      scala.math.exp(mainPreact)/total
    }
      
    def derivative(n : OutNeuron, target : Int) : Double = n match {
      case x : OutNeuron if x.classification==target => 1.0 - x.act
      case _ => 0.0 - n.act
    }
  }
  