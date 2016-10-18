package cnn.learning

import cnn.core.structure.{Network, InputLayer}
import cnn.core.structure.NonEmptyMat
import cnn.core.structure.FCLayer

class LearnSession(set : Seq[Example], net : Network) {
  
  def doTrain = {
    println("The training phase beguin...")
    train(set, net)
  }
  
  
  def doTest = {
    println("The test phase beguin...")
    test(set, Seq(), net)
  }
  
  
  private def test (set : Seq[Example], acc : Seq[Example] , net : Network) : Seq[Example] = set match {
    case x +: xs => val res = net.submit(x)
                                 .compute
                    val inf = res.getInference
                    println("Result : "+ inf._1+ "target is : "+ x.classification)
                    res.last.asInstanceOf[FCLayer].neurons.foreach { x => print(x._act+",")}
                    println()
                    val n = if(inf._1 == x.classification) x.updateFlag(true) else x
                    test(xs, acc :+ n, net)
    case _ => acc
  }
  
  @scala.annotation.tailrec
  private def train(set : Seq[Example], net : Network) : Network = set match {
      case a@ x +: xs if(a.forall { x => x.learned }) => net
      case x +: xs => val res = net.submit(x)
                                   .compute
                      val inf = res.getInference
                      /**Debug**/
                      println("Iteration nb : "+ res.lc.iteration + ", result : "+ inf._1 +" target is : "+ x.classification)
                      res.last.asInstanceOf[FCLayer].neurons.foreach { x => print(x._act+",")}
                      println()
                      /**Debug**/
                      if(inf._1 == x.classification && inf._2 > 0.7)
                        train(xs.:+(x.updateFlag(true)), net)
                      else train(xs.:+(x.updateFlag(false)), res.backPropagation.updateWeight)
                      
      case Seq()  => throw new IllegalArgumentException("The given training set is empty")
    }
  

}