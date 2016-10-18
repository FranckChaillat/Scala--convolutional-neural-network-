package cnn.helper.implicits

import scala.annotation.tailrec


object Implicits {
  
  
    case class AccumulatorParam[A](base : Seq[A]) {
      def add(a: A) = base.:+(a)
    }
    

    
    implicit class RichInt(i : Int){
      
      def gcd(a : Int) : Int =  
        Range(1,i).filter(x=> a*x ==i) match {
              case IndexedSeq() => i
              case l @ x +: _ => l.max
       }
      
      def isPrime = (i%2 != 0) && (i%3 !=0) && (i%4!=0) && (i%5!=0)
      def isBetween(min : Double, max : Double) = i >= min && i < max
      
    }
    
    
    implicit class RichList[A](l : Vector[A]){
      
      def combineAll(l2 : Vector[A]) = l.flatMap( x => l2.map( y => (x,y)))
     
      
    }
    
    object RichList{
       def powerZip[A](l : Vector[Vector[A]]) = {
         Range(0, l.minBy (_.size).size).map(j=> l.map(e=> e(j))).toVector
      }
    }
   
}