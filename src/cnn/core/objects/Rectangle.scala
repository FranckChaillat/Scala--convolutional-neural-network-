package cnn.core.objects

 case class Rectangle(x : Int, y: Int, width : Int, heigh : Int){
  
    def getRectWidth() = width - x
    
    def getRectHeigh() = heigh - y
  
    def goRight() : Option[Rectangle] = Option.apply( Rectangle(x+1, y, width+1 ,heigh))
    
    def goRightNonOverlap() : Option[Rectangle] =  Option.apply( Rectangle(x+(width-x), y, width.+(width-x), heigh))
    
    def lineDown() : Option[Rectangle]= Option.apply( Rectangle(0, y+1, width - x, (y+1) + (heigh - y)))
    
    def lineDownNonOverlap() : Option[Rectangle]= {
      val h=  heigh-y
      Option(Rectangle(0, y+h, width - x, (y+h) + (heigh - y)))
    }
    
    def lineDownFull() : Option[Rectangle] = {
      val x = 1-getRectWidth
      Option(Rectangle(x, y+1,x+ getRectWidth ,heigh+1))
    }
    
    def isOutOfBound(limitW: Int, limitH : Int) = width > limitW || heigh > limitH
    
   
  }