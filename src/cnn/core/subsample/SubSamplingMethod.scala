package cnn.core.subsample


sealed trait SubSamplingMethod { def value : String}
case object _MAXPOOLING extends SubSamplingMethod { val value = "1"}
case object _MEANPOOLING extends SubSamplingMethod { val value = "2"}