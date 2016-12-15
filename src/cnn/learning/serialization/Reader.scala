package cnn.learning.serialization

import scala.xml.XML
import scala.xml.{NodeSeq, Node, Elem}
import java.io.File
import scala.io.Source
import cnn.exceptions.{NetworkLoadingException, NOT_A_DIRECTORY, XML_CONTENT_ERR}
import cnn.core.structure.NonEmptyMat
import cnn.core.structure.Kernel
import cnn.activation.functions._
import cnn.exceptions.NetworkLoadingException
import cnn.core.structure.ConvolutionLayer
import cnn.core.structure._
import cnn.core.subsample.{_MAXPOOLING, _MEANPOOLING}
import cnn.exceptions.NetworkLoadingException
import cnn.learning.LearningContext



object Reader {
  
    def apply(path : String) = {
      val content = if(!new File(path).exists) throw NetworkLoadingException(NOT_A_DIRECTORY)
      else scala.xml.Utility.trim( XML.loadString( Source.fromFile(path).getLines.mkString))
      
      
      try {
        
         val layers = content.\\ ("network").\\("layers").head.child
                             .map(x=> loadLayer(x))
         val lc = new LearningContext(0,0,content.\\("lc").text.toDouble)
         Some(new Network(layers.toVector, lc))
      }catch{
        case e : Throwable => None
      }
        
    }
    
    def loadLayer(layer : Node) : Layer[NeuralUnit]= layer match {
      case <conv>{kernels, fun}</conv> => val kern = kernels.\\("kernel")
                                                    .map(k=>  k.\\("value").headOption
                                                    .fold(throw NetworkLoadingException(XML_CONTENT_ERR))(_.text ))
                                                    .map(x=> { val value = x.split("/")
                                                                            .map(x=> x.split(";").toVector.map(_.toDouble)).toVector
                                                              Kernel(value, functionFromXml(fun))
                                                       })
                                         ConvolutionLayer(kern.toVector)
      
      case <pool>{window, method}</pool> => val win = window.text.split(";")
                                            val meth = method.text match {
                                              case "MAXPOOLING" => _MAXPOOLING
                                              case "MEANPOOLING" => _MEANPOOLING
                                           }
                                     PoolingLayer(new PoolingWindow(win(0).toInt, win(1).toInt), meth)
                                     
      case <fc>{neurons, fun}</fc>   => val neur =  neurons.child.map(x=> {
                                          val weight = x.\\("weight").text.split(";").map(x=> new Link(x.toDouble)).toVector
                                          val classification  = if(x.child.exists(e => e.label == "class")) Some(x.\\("class").text) else None
                                          classification.fold(new Neuron(weight, functionFromXml(fun))) (x => new OutNeuron(weight, x.toInt))
                                       })
                                     FCLayer(neur.toVector)
      
      case _ => throw NetworkLoadingException(XML_CONTENT_ERR)
    }
    
    private def functionFromXml(fun : Node) = fun match {
          case <fun>SIGMOID</fun> => _SIGMOID
          case <fun>SOFTMAX</fun> => _SOFTMAX
          case _ => throw NetworkLoadingException(XML_CONTENT_ERR)
    }
    
}