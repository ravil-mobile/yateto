from ..common import *

####################################################################################################
#                                         CPU
####################################################################################################
class Generic(object):
  def __init__(self, arch, descr):
    self._arch = arch
    self._descr = descr


  def _formatTerm(self, alpha, term):
    """Generate a sub-string of a term for a source code which is going to be used
    inside of the inner most for-loop

    Args:
      alpha (Union[Scalar, float]): TODO
      term (IndexedTensorDescription): TODO

    Returns:

    Examples:
      >>> from yateto.memory import DenseMemoryLayout
      >>> from yateto.ast.indices import Indices
      >>> from yateto.codegen.common import IndexedTensorDescription
      >>> from yateto.aspp import dense
      >>> from yateto.codegen.copyscaleadd.generic import Generic
      >>> tensor_shape = (5, 6)
      >>> layout = DenseMemoryLayout(shape=tensor_shape)
      >>> indices = Indices(indexNames='ij', shape=tensor_shape)
      >>> description = IndexedTensorDescription(name='A', \
                                                 indices=indices, \
                                                 memoryLayout=layout, \
                                                 eqspp=dense(shape=tensor_shape))
      >>> obj = Generic(arch='dummy', descr=description)
      >>> obj._formatTerm(alpha=3, term=description)
      '3 * A[1*i + 5*j]'

    """

    prefix = ''
    if alpha == 0.0:
      return ''

    if alpha == 1.0:
      prefix = term.name
    else:
      prefix = '{} * {}'.format(alpha, term.name)

    return '{}[{}]'.format(prefix, term.memoryLayout.addressString(term.indices))


  def generate(self, cpp, routineCache):
    """Generates a tensor equation of a form: B = beta * B + alpha * A
    Args:
      cpp (IO): a file stream
      routineCache:

    Returns:

    """
    description = self._descr  # type: copyscaleadd.Description
    
    if description.beta == 0.0:
      writeBB = boundingBoxFromLoopRanges(description.result.indices, description.loopRanges)
      initializeWithZero(cpp, self._arch, description.result, writeBB)

    class CopyScaleAddBody(object):
      def __call__(s):

        operation = '='
        flop = 0
        alpha = description.alpha

        if alpha not in [-1.0, 1.0]:
          flop += 1

        if description.beta == 1.0 and alpha == -1.0:
          alpha = 1.0
          operation = '-='
          flop += 1

        elif description.beta == 1.0:
          operation = '+='
          flop += 1

        elif description.beta != 0.0:
          raise NotImplementedError

        cpp('{} {} {};'.format(self._formatTerm(1.0, description.result),
                               operation,
                               self._formatTerm(alpha, description.term)))

        return flop

    return forLoops(cpp, description.result.indices, description.loopRanges, CopyScaleAddBody())


####################################################################################################
#                                         CUDA
####################################################################################################
from yateto.type import Tensor
class GenericCuda(object):
  def __init__(self, arch, descr):
    self._arch = arch
    self._descr = descr

  def _formatTerm(self, alpha, term):
    """Generate a sub-string of a term for a source code which is going to be used
    inside of the inner most for-loop

    Args:
      alpha (Union[Scalar, float]): TODO
      term (IndexedTensorDescription): TODO

    Returns:

    Examples:
      TODO: update example
      >>> from yateto.memory import DenseMemoryLayout
      >>> from yateto.ast.indices import Indices
      >>> from yateto.codegen.common import IndexedTensorDescription
      >>> from yateto.aspp import dense
      >>> from yateto.codegen.copyscaleadd.generic import Generic
      >>> tensor_shape = (5, 6)
      >>> layout = DenseMemoryLayout(shape=tensor_shape)
      >>> indices = Indices(indexNames='ij', shape=tensor_shape)
      >>> description = IndexedTensorDescription(name='A', \
                                                 indices=indices, \
                                                 memoryLayout=layout, \
                                                 eqspp=dense(shape=tensor_shape))
      >>> obj = Generic(arch='dummy', descr=description)
      >>> obj._formatTerm(alpha=3, term=description)
      '3 * A[1*i + 5*j]'

    """

    prefix = ''
    if alpha == 0.0:
      return ''

    if alpha == 1.0:
      prefix = term.name
    else:
      prefix = '{} * {}'.format(alpha, term.name)

    return '{}[{}]'.format(prefix, term.memoryLayout.addressString(term.indices))

  def generate(self, cpp, routineCache):
    """Generates a tensor equation of a form: B = beta * B + alpha * A
    Args:
      cpp (IO): a file stream
      routineCache:

    Returns:

    """


    description = self._descr  # type: copyscaleadd.Description

    if description.beta == 0.0:
      #TODO: figure out how to do this trick on gpu
      #writeBB = boundingBoxFromLoopRanges(description.result.indices, description.loopRanges)
      #initializeWithZero(cpp, self._arch, description.result, writeBB)
      pass



    # extract first two leading tensor indices
    # summation of which is going to be computed on GPU
    leading_indices = (description.result.indices[0],
                       description.result.indices[1])   # type: Set[str]

    self.cuda_kernel_indices = \
      description.result.indices.extract(indexNames="".join(leading_indices))  # type: Set[Indices]


    for_loop_index_names = description.result.indices - self.cuda_kernel_indices  # type: Set[str]
    for_loop_indices = \
      description.result.indices.extract(indexNames="".join(for_loop_index_names))

    class CopyScaleAddBody(object):
      def __call__(s):


        flop = 0

        parameters = "{}, {}, ".format(description.loopRanges[self.cuda_kernel_indices[0]].stop,
                                       description.loopRanges[self.cuda_kernel_indices[1]].stop)


        result_data_shift = 0
        term_data_shift = 0
        for i in range(len(description.result.indices)):
          result_data_shift += description.result.memoryLayout.stridei(i) \
                               * description.loopRanges[description.result.indices[i]].start

          term_data_shift += description.term.memoryLayout.stridei(i) \
                             * description.loopRanges[description.term.indices[i]].start

        # append indices w.r.t outer-most (for-loop) loop indices
        if len(description.result.indices) > len(self.cuda_kernel_indices):

          result_data_shift = "{} + {}".format(str(result_data_shift),
                                               description.result.memoryLayout.addressString(description.result.indices, for_loop_indices))

          term_data_shift = "{} + {}".format(str(term_data_shift),
                                            description.term.memoryLayout.addressString(description.term.indices, for_loop_indices))


        parameters += "{}, {} + {}, {}, ".format(description.alpha,
                                                 description.term.name,
                                                 term_data_shift,
                                                 description.term.memoryLayout.stridei(1))

        parameters += "{}, {} + {}, {}, ".format(description.beta,
                                                 description.result.name,
                                                 result_data_shift,
                                                 description.result.memoryLayout.stridei(1))

        if Tensor.getGroup(description.term.name):
          term_group = "[{}]".format(*Tensor.getGroup(description.term.name))
        else:
          term_group = ""

        term_base_name = Tensor.getBaseName(description.term.name)
        parameters += "tensor::{}::jump_to_next{}, ".format(term_base_name,
                                                            term_group)


        if Tensor.getGroup(description.result.name):
          result_group = "[{}]".format(*Tensor.getGroup(description.result.name))
        else:
          result_group = ""

        result_base_name = Tensor.getBaseName(description.result.name)
        parameters += "tensor::{}::jump_to_next{}, ".format(result_base_name,
                                                            result_group)

        parameters += "tensor::num_elements_in_cluster"

        cpp('cuda_copy_add_scale({});'.format(parameters))

        """
        alpha = description.alpha

        if alpha not in [-1.0, 1.0]:
          flop += 1

        if description.beta == 1.0 and alpha == -1.0:
          alpha = 1.0
          operation = '-='
          flop += 1

        elif description.beta == 1.0:
          operation = '+='
          flop += 1

        elif description.beta != 0.0:
          raise NotImplementedError

        cpp('{} {} {};'.format(self._formatTerm(1.0, description.result),
                               operation,
                               self._formatTerm(alpha, description.term)))
        """

        return flop


    return forLoops(cpp=cpp,
                    indexNames=for_loop_indices,
                    ranges=description.loopRanges,
                    body=CopyScaleAddBody(),
                    pragmaSimd=False)
