package cnn.core.convolution

sealed trait ConvolutionMethod { def value : String}
case object _SAME extends ConvolutionMethod { val value = "1"}
case object _VALID extends ConvolutionMethod { val value = "2"}
case object _FULL extends ConvolutionMethod { val value = "3"}