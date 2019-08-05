from ..common import *
from .generic import Generic, GenericCuda

class Description(object):
  """The class holds all necessary incormation in order to compute an equation in a form:
  B = beta * B + alpha * A
  """

  def __init__(self, alpha, beta, result, term):
    """
    Args:
      alpha (Union[Scalar, float]): a term multiplier
      beta (float): decides whether to use the first term of rhs or not (0.0 or 1.0 supported)
      result (IndexedTensorDescription): a description of rhs
      term (IndexedTensorDescription): a description of lhs (second term)
    """
    self.alpha = alpha
    self.beta = beta
    self.result = result
    self.term = term
    
    assert self.alpha != 0.0, 'copyscaleadd does not support alpha=0.0 at the moment.'
    assert self.beta == 1.0 or self.beta == 0.0, 'copyscaleadd supports only beta=0.0 or beta=1.0 at the moment.'
    
    assert self.result.indices == self.term.indices

    term_range_table = loopRanges(self.term, self.term.indices)
    result_range_table = loopRanges(self.result, self.result.indices)

    # check whether ranges of the term tensor contain ranges of the result of a computation
    # NOTE: a range is a span of values of a tensor index
    assert testLoopRangesAContainedInB(term_range_table, result_range_table)
    
    self.loopRanges = term_range_table  # type: Dict[str, Range]
    

def generator(arch, descr):
  return Generic(arch, descr)


def produce_cuda_generator(arch, descr):
  return GenericCuda(arch, descr)

