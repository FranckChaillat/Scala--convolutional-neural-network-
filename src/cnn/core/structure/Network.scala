package cnn.core.structure

import cnn.core.convolution.Convolution._
import cnn.core.subsample.SubSampling._
import scala.annotation.tailrec
import cnn.exceptions. {MatCountException,InvalidNeuralUnitTypeException,LayerTypeException,NetworkStructureException, KERNEL_UPDATE_INPUT}
import cnn.learning.LearningContext
import cnn.exceptions.MatTypeException
import cnn.exceptions.NeuronTypeException
import cnn.exceptions._
import cnn.learning.Example



 class Network(val layers : Vector[Layer[NeuralUnit]], val lc : LearningContext){
  
  
  def this() = this(Vector(), LearningContext(0,0,0))
  def this(lc : LearningContext) = this(Vector(), lc)
  def :+[A <: NeuralUnit](l: Layer[A]): Network =  new Network( layers.:+(l), lc)
  def +:(l : Layer[NeuralUnit]): Network = new Network(layers.+: (l), lc)
  def count = layers.size
  def last = layers.last
  def first = layers.head
  def apply(i : Int) = layers(i)
  
  def getInputLayer = layers.headOption match {
      case Some(a) => a match {
        case in : InputLayer => in
        case _ => throw LayerTypeException(NO_INPUT_LAYER)
      }
    
    case None => throw NetworkStructureException(NO_INPUT_LAYER)
  }
  

  def submit(ex : Example) = ex.m match {
    case ne : NonEmptyMat => 
      val in = InputLayer(Vector(ne))
      new Network(layers.tail.+:(in), lc.updateTarget(ex.classification))
    case _ => throw MatTypeException(EMPTY_MAT)
  }
  
  def getInference = layers.lastOption.fold(throw NetworkStructureException(EMPTY_NETWORK)){_ match {
    case fc : FCLayer => val max = fc.get.collect{ case x: OutNeuron => x
                                                   case _ => throw NeuronTypeException(NO_OUTPUT_LAYER)
                                                 }
                                         .maxBy(_.act)
                        (max.classification, max.act)
    case _ => throw NetworkStructureException(NOT_FC)
  }}
  
  
  def compute() : Network = {
    val inputLayer = getInputLayer
    val res =_compute(new Network( Vector(inputLayer), lc), layers.tail)
    res.lc.iteration+=1
    res
  }
   
  
   @tailrec
   private def _compute(acc : Network , l : Vector[Layer[NeuralUnit]]) : Network = l.headOption match {
        case None => acc
        case Some(layer) => layer match {
          /*if next layer is Pooling*/
          case pl : PoolingLayer =>
            val in = acc.last match {
              case p : PoolingLayer     => p.getActivation.get
              case c : ConvolutionLayer => c.getActivation.get
              case i : InputLayer if acc.last == acc.first => InputLayer(i.get)
              case _ => throw LayerTypeException(INVALID_LAYER_ORDER)
            }
              _compute(acc.:+(pl(in)), l.tail)
          
          /*if next layer is Convolutive*/
          case cl : ConvolutionLayer => 
            val in = acc.last match {
              case p : PoolingLayer => p.getActivation.get
              case c : ConvolutionLayer => c.getActivation.get
              case i : InputLayer if acc.last == acc.first => InputLayer(i get)
              case _ => throw LayerTypeException(INVALID_LAYER_ORDER)
            }
            _compute(acc.:+(cl(in)), l.tail)
          
          /*if next layer is FullyConnected*/  
          case fc : FCLayer  => 
           val in = acc.last match {
              case c : ConvolutionLayer =>  Layer(c.getActivation.get.get.flatMap(x => Mat.toNeurons(x)))
              case p : PoolingLayer =>  Layer(p.getActivation.get.get.flatMap(x=> Mat.toNeurons(x)))
              case f : FCLayer =>  Layer(f.get)
            }
           _compute(acc.:+(fc(in)), l.tail)
          
        }
   }
  
  def backPropagation() = {
    
      @tailrec
      def backprop(pLayers : Vector[Layer[NeuralUnit]], acc : Network = new Network(lc)) : Network = {
    
        pLayers.reverse match {
          case h +: t if h == layers.last => h match {
            case a : ProcessableLayer[_] => backprop(t.reverse, acc.+:(a.derivate(None, lc))) 
            case _ => throw NetworkStructureException(PROCESSABLE_LAYER_ONLY)
          }
          case h +: t => (h, acc.first) match {
                case (a : ProcessableLayer[_], b : ProcessableLayer[_]) => backprop(t.reverse, acc.+:(a.derivate(Option(b), lc) ))
                case _ => throw  NetworkStructureException(PROCESSABLE_LAYER_ONLY)
          }
          case _ => acc
        }
     }
    
    layers match {
      case x @ h +:t => h match {
        case a : ProcessableLayer[_] => throw NetworkStructureException(NO_INPUT_LAYER)
        case _ => h +: backprop(t)
      }
      case _ => throw NetworkStructureException(EMPTY_NETWORK)
    }
  } 
  
  def updateWeight() = {
    
    @tailrec
    def update(l : Vector[Layer[NeuralUnit]], acc : Network) : Network = (l.headOption, l.tail.headOption) match {
      case (Some(cur), Some(nxt)) => (cur, nxt) match {
        case (conv : ConvolutionLayer, x : ProcessableLayer[_]) => update(l.tail, acc.:+(conv.updateKernel(x, lc)))
        case (x, y) => update(l.tail, acc.:+(x))
      }
      case (Some(cur), None) => cur match {
        case fc : FCLayer => acc.:+(fc.updateWeight(lc))
        case _ => throw NetworkStructureException(INVALID_OUT_LAYER)
      }
      case _ => throw NetworkStructureException(GEN_INVALID_LAYER)
   }
    
    update(layers, new Network(lc))

  }
    
}
   



  
     
