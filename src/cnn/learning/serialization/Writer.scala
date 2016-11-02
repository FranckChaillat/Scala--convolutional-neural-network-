package cnn.learning.serialization

import cnn.core.structure._
import scala.xml.{Node, NodeSeq}
import scala.xml.XML
import scala.xml.Elem
import scala.concurrent.Future
import scala.util.Success
import scala.util.Failure
import scala.annotation.tailrec


object Writer {

  
  def apply(path : String, net : Network) = {
    
     
     def record(acc : Seq[Elem], l : Seq[Layer[NeuralUnit]]) : Seq[Elem] = l.headOption match {
       case None => acc
       case Some(x) => x match {
               case ConvolutionLayer(k) if k.size >0 && k.forall(_.activationFunc == k.head.activationFunc ) =>
                    val kernelsXml = k.map(x=> <kernel><value>{x.get.map(_ mkString(";")) mkString("/")}</value></kernel>)  
                    record(acc :+ <conv>
																			<kernels>{kernelsXml}</kernels>
                  										<fun>{k.head.activationFunc.value}</fun>
                  							</conv>, l tail)
               
               case  PoolingLayer(w,m) => val poolingLayerXml = <pool>
																            <window>{w.x+";"+w.y}</window>
                                            <method>{m.value}</method>
															            </pool>
                    record(acc :+ poolingLayerXml, l tail)
               
               case  FCLayer(n) if n.size >0 && n.forall(_._activationFun == n.head._activationFun) => 
                    val fcXml = <fc> <neurons>{ n.map(x=> <neuron>
																													<weight>{x._inLinks.map(_.weight).mkString(";")}</weight>
																												  { x match { case o : OutNeuron => <class>{o.classification}</class> 
                                                                      case _ => Node.EmptyNamespace
                                                                    }
                                                          }
																												</neuron>) }</neurons>
										           		<fun>{n.head._activationFun.value}</fun>
													     </fc>
                    record(acc :+ fcXml, l tail)
               
               case _ => record(acc, l tail)
        }
     }
     
     val netXml = <network>
								 <layers>{record(Seq(), net.layers)}</layers>
								 <lc>{net.lc.leaningRate}</lc>
	 							</network>
                   
                   
     import scala.concurrent.ExecutionContext.Implicits.global
     Future{
      val trimed = scala.xml.Utility.trim(netXml)
      XML.save(path, trimed)
     }.andThen{
       case Success(s) => println("the network have succefully been written to"+path)
                          s
       case Failure(f) => throw f
     }             
  
    }
}