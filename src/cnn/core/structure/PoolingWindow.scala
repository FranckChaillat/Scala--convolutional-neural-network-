package cnn.core.structure

import cnn.core.structure.Layer._


class PoolingWindow(val x : Int, val y : Int, val input : Layer[NonEmptyMat] , val activation : Layer[NonEmptyMat], val delta : Layer[NonEmptyMat]) {
  def this(x : Int, y: Int){
    this(x, y, Layer.getEmpty(), Layer.getEmpty(), Layer.getEmpty())
  }
  
  def this(x : Int, y : Int, in : Layer[NonEmptyMat]){
    this(x, y, in, Layer.getEmpty(), Layer.getEmpty())
  }
  
  def this(x : Int, y : Int, in : Layer[NonEmptyMat], act : Layer[NonEmptyMat]) {
    this(x,y,in,act,Layer.getEmpty())
  }
}

object PoolingWindow{
    def unapply(p : PoolingWindow) : Option[(Int, Int, Layer[NonEmptyMat], Layer[NonEmptyMat])] = Some(p.x,p.y,p.input,p.activation)
    
}