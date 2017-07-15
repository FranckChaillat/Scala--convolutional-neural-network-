package cnn.learning

import cnn.core.structure.{Network, InputLayer}
import cnn.core.structure.NonEmptyMat
import cnn.core.structure.FCLayer
import scala.annotation.tailrec


object LearnSession {
  

  def test = {
      @tailrec
      def _test(acc: Seq[Example]) (set: Seq[Example], net: Network) : Seq[Example] = set match {
          case x +: xs => val res = net.submit(x)
                                       .compute
                          val inf = res.getInference
                          println("Result : "+ inf._1+ "target is : "+ x.classification)
                          res.last.asInstanceOf[FCLayer].neurons.foreach { x => print(x._act+",")}
                          println()
                          val n = if(inf._1 == x.classification) x.updateFlag(true) else x
                          _test(acc :+ n)(xs, net)
          case _ => acc
      }
     _test(Seq())_
  }
  
  
  @tailrec
    def train(set : Seq[Example], net : Network) : Network = set match {
      case a@ x +: xs if(a.forall { x => x.learned }) => net
      case x +: xs => 
                      val res = net.submit(x)
                                   .compute
                      val inf = res.getInference
                      
                      /**Debug**/
                      println("Iteration nb : "+ res.lc.iteration + ", result : "+ inf._1 +" target is : "+ x.classification)
                      res.last.asInstanceOf[FCLayer].neurons.foreach { x => print(x._act+",")}
                      println()
                      /**Debug**/
                      if(inf._1 == x.classification && inf._2 > 0.7){
                        println("OK")
                        train(xs.:+(x.updateFlag(true)), net)
                      }
                      else
                      {
                        println("KO")
                        train(xs.:+(x.updateFlag(false)), res.backPropagation.updateWeight)
                      }
                      
      case Seq()  => throw new IllegalArgumentException("The given training set is empty")
    }


}