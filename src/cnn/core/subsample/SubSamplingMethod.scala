package cnn.core.subsample


sealed trait SubSamplingMethod { def value : String}
case object _MAXPOOLING extends SubSamplingMethod { val value = "MAXPOOLING"}
case object _MEANPOOLING extends SubSamplingMethod { val value = "MEANPOOLING"}