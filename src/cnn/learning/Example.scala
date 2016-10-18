package cnn.learning

import cnn.core.structure.Mat

case class Example(classification : Int, m : Mat, learned : Boolean = false){
  def updateFlag(b : Boolean) = Example(classification, m, b)
}

