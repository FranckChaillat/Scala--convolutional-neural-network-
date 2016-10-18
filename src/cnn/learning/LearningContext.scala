package cnn.learning

case class LearningContext(var target : Int, var iteration : Int, leaningRate : Double){
    def updateTarget(t : Int) = LearningContext(t, iteration, leaningRate)
}