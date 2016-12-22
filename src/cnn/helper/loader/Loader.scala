package cnn.helper.loader

import java.io.File
import cnn.exceptions.{ExempleLoadException, NOT_A_DIRECTORY}
import cnn.core.structure.NonEmptyMat
import org.opencv.core.Core
import org.opencv.highgui.Highgui
import cnn.learning.Example

object Loader {

  def load(dirPath : String) = {
    
    val f = new File(dirPath)
    if(f.exists && f.isDirectory){
      
      System.loadLibrary(Core.NATIVE_LIBRARY_NAME)
      
      f.listFiles
       .filter(x => x.isDirectory)
       .flatMap(_.listFiles)
       .filter(x=> x.getName.endsWith(".jpg")||
                   x.getName.endsWith(".png"))
       .map (x => {
                    val gray = Highgui.imread(x.getAbsolutePath, Highgui.CV_LOAD_IMAGE_GRAYSCALE)
                    val t = (for(i <- Range(0, gray.rows)) yield
                                  (for(j <- Range(0, gray.cols)) yield
                                    gray.get(i, j)(0)).toVector).toVector
                    
                    Example(x.getParentFile.getName.toInt, new NonEmptyMat(t)) 
             })

       
       
      
    }
    else throw ExempleLoadException(NOT_A_DIRECTORY)
  }
}