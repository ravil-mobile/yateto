import string
from ..ast.indices import Indices, Range
from ..ast.node import IndexedTensor
from ..memory import DenseMemoryLayout
from .common import forLoops, TensorDescription, IndexedTensorDescription
from . import copyscaleadd, indexsum, log, product
from functools import reduce


class CudaKernelFactory(object):
  ERROR_NAME = '_error'

  def __init__(self, cpp, arch):
    """
    Args:
      cpp (TODO): a file descriptor
      arch (Architecture): TODO
    """
    self._cpp = cpp
    self._arch = arch
    self._freeList = list()


  def create(self, node, *args):
    method = 'create_' + node.__class__.__name__
    factory = getattr(self, method, self.generic_create)
    return factory(node, *args)


  def generic_create(self, node, *args):
    raise NotImplementedError


  def simple(self, result, term, add, scalar, routineCache):
    raise NotImplementedError


  def temporary(self, buffer_name, buffer_size, iniZero=False, memory=list()):
    """
    Allocates memory on CPU
    Args:
      buffer_name (str): buffer name
      buffer_size (int): size of buffer
      iniZero:
      memory:

    Returns:

    """

    assert(iniZero == False or len(memory) == 0)
    if self._arch.onHeap(buffer_size):
      if memory:
        raise NotImplementedError('Direct temporary initialization is not supported for heap-allocated memory.')
      if len(self._freeList) == 0:
        self._cpp('int {};'.format(self.ERROR_NAME))
      self._cpp('{}* {};'.format(self._arch.typename, buffer_name))
      self._cpp('{} = posix_memalign(reinterpret_cast<void**>(&{}), {}, {}*sizeof({}));'.format(
                  self.ERROR_NAME,
                  buffer_name,
                  self._arch.alignment,
                  buffer_size,
                  self._arch.typename))
      if iniZero:
        self._cpp.memset(buffer_name, buffer_size, self._arch.typename)
      self._freeList.append(buffer_name)
    else:
      ini = ''
      if iniZero:
        ini = ' = {}'
      elif memory:
        ini = ' = {{{}}}'.format(', '.join(memory))
      self._cpp('// allocating temp memory only on cpu')
      self._cpp('{} {}[{}] __attribute__((aligned({}))){};'.format(self._arch.typename,
                                                                   buffer_name,
                                                                   buffer_size,
                                                                   self._arch.alignment,
                                                                   ini))
 

  def cuda_temporary(self, buffer_name, buffer_size, iniZero=False, memory=list()):
    """
    Allocates memory for a buffer on GPU i.e. for scratch memory
    Args:
      buffer_name (str): buffer name
      buffer_size (int): size of buffer
      iniZero:
      memory:

    Returns:

    """

    assert(iniZero == False or len(memory) == 0)

    self._cpp('// allocating temp memory only on gpu')
    self._cpp('{0} *{1};'.format(self._arch.typename, buffer_name))

    total_buffer_volume = "{} * yateto::tensor::num_elements_in_cluster".format(buffer_size)
    self._cpp('cudaMalloc(&{0}, sizeof({1}) * {2}); CUDA_CHECK;'.format(buffer_name,
                                                                        self._arch.typename,
                                                                        total_buffer_volume))

  def cuda_delete_temporary(self, buffer_name):
    self._cpp("cudaFree({}); CUDA_CHECK;".format(buffer_name))
    pass


  def freeTmp(self):
    for free in self._freeList:
      self._cpp('free({});'.format(free))
    self._freeList = []


  def _indices(self, var):
    shape = var.memoryLayout().shape()
    return Indices(string.ascii_lowercase[:len(shape)], shape)


class CudaOptimisedKernelFactory(CudaKernelFactory):
  def __init__(self, cpp, arch):
    """
    Args:
      cpp (TODO): a file descriptor
      arch (Architecture): TODO
    """
    super().__init__(cpp, arch)

  def create_LoopOverGEMM(self,
                          node,
                          result,
                          arguments,
                          add,
                          scalar,
                          prefetchName,
                          routineCache,
                          gemm_cfg):
    """TODO

    Args:
      node (Type[Node]):
      result (Variable):
      arguments (List[Varaible]):
      add (bool):
      scalar (Union[float, Scalar, None]):
      prefetchName (Union[str, None]):
      routineCache (RoutineCache):
      gemm_cfg (GeneratorCollection):

    Returns:
      TODO:
    """
    assert len(arguments) == 2
    description = log.Description(
      alpha=scalar,
      add=add,
      result=IndexedTensorDescription.fromNode(result, node),
      leftTerm=IndexedTensorDescription.fromNode(arguments[0], node.leftTerm()),
      rightTerm=IndexedTensorDescription.fromNode(arguments[1], node.rightTerm()),
      loopIndices=node.loopIndices(),
      transA=node.transA(),
      transB=node.transB(),
      prefetchName=prefetchName,
      is_cuda_factory_used=isinstance(self, CudaKernelFactory)
    )

    generator = log.generator(self._arch, description)
    return generator.generate(self._cpp, routineCache, gemm_cfg)


  def create_IndexSum(self, node, result, arguments, add, scalar, prefetchName, routineCache, gemm_cfg):
    assert len(arguments) == 1
    description = indexsum.Description(alpha=scalar,
                                       add=add,
                                       result=IndexedTensorDescription.fromNode(result, node),
                                       term=IndexedTensorDescription.fromNode(arguments[0], node.term())
    )
    generator = indexsum.generator(self._arch, description)
    return generator.generate(self._cpp, routineCache)
  
  def create_Product(self, node, result, arguments, add, scalar, prefetchName, routineCache, gemm_cfg):
    assert len(arguments) == 2
    description = product.Description(
      alpha=scalar,
      add=add,
      result=IndexedTensorDescription.fromNode(result, node),
      leftTerm=IndexedTensorDescription.fromNode(arguments[0], node.leftTerm()),
      rightTerm=IndexedTensorDescription.fromNode(arguments[1], node.rightTerm())
    )
    generator = product.generator(self._arch, description)
    return generator.generate(self._cpp, routineCache)


  def simple(self, result, term, add, scalar, routineCache):
    if scalar and scalar != 1.0:

      # compute the total volume of tensors to multiply by a scalar
      # TODO: this part of the code must be generilized
      # in case of the user wants to perform scalar-tensor
      # multiplication with just one tensor

      tensor_volume = term.memoryLayout().requiredReals()
      tensor_volume_as_str = "{} * tensor::num_elements_in_cluster".format(tensor_volume)


      if add:
        # generate cuda call for scalar tensor product
        self._cpp('cuda_scalar_tensor_mult_add({}, {}, {}, {});'.format(scalar,
                                                                        term.name,
                                                                        result.name,
                                                                        tensor_volume_as_str))
      else:
        # generate cuda call for scalar tensor product
        self._cpp('cuda_scalar_tensor_mult({}, {}, {}, {});'.format(scalar,
                                                                    term.name,
                                                                    result.name,
                                                                    tensor_volume_as_str))
      self._cpp.emptyline()
    # TODO: return a number of flop
    return 0

class CudaUnitTestFactory(CudaKernelFactory):
  def __init__(self, cpp, arch, nameFun):
    super().__init__(cpp, arch)
    self._name = nameFun


  def _formatTerm(self, var, indices):
    address = var.memoryLayout().addressString(indices)
    return '{}[{}]'.format(self._name(var), address)


  def create_Einsum(self, node, result, arguments, add, scalar, prefetchName, routineCache, gemm_cfg):
    g = node.indices
    for child in node:
      g = g.merged(child.indices - g)
    
    ranges = {idx: Range(0, g.indexSize(idx)) for idx in g}
    
    resultTerm = self._formatTerm(result, node.indices)
    terms = [self._formatTerm(arguments[i], child.indices) for i,child in enumerate(node)]
    
    if scalar and scalar != 1.0:
      terms.insert(0, str(scalar))
    
    if not add:
      self._cpp.memset(self._name(result), result.memoryLayout().requiredReals(), self._arch.typename)
    
    class EinsumBody(object):
      def __call__(s):
        self._cpp( '{} += {};'.format(resultTerm, ' * '.join(terms)) )
        return len(terms)

    return forLoops(self._cpp, g, ranges, EinsumBody(), pragmaSimd=False)


  def create_ScalarMultiplication(self, node, result, arguments, add, scalar, prefetchName, routineCache, gemm_cfg):
    return self.simple(result, arguments[0], add, scalar, routineCache)


  def simple(self, result, term, add, scalar, routineCache):
    g = self._indices(result)

    ranges = {idx: Range(0, g.indexSize(idx)) for idx in g}

    resultTerm = self._formatTerm(result, g)
    termTerm = self._formatTerm(term, g)

    if scalar and scalar != 1.0:
      termTerm = '{} * {}'.format(scalar, termTerm)

    class AssignBody(object):
      def __call__(s):
        self._cpp('{} {} {};'.format(resultTerm, '+=' if add else '=', termTerm))
        return 1 if add else 0

    return forLoops(self._cpp, g, ranges, AssignBody(), pragmaSimd=False)


  def compare(self, ref, target, epsMult = 100.0):
    g = self._indices(ref)
    refTerm = self._formatTerm(ref, g)
    targetTerm = self._formatTerm(target, g)

    class CompareBody(object):
      def __call__(s):
        self._cpp('double ref = {};'.format(refTerm))
        self._cpp('double diff = ref - {};'.format(targetTerm))
        self._cpp('error += diff * diff;')
        self._cpp('refNorm += ref * ref;')
        return 0

    targetBBox = target.memoryLayout().bbox()
    ranges = {idx: Range(targetBBox[i].start, min(targetBBox[i].stop, g.indexSize(idx))) for i,idx in enumerate(g)}
    with self._cpp.AnonymousScope():
      self._cpp('double error = 0.0;')
      self._cpp('double refNorm = 0.0;')
      forLoops(self._cpp, g, ranges, CompareBody(), pragmaSimd=False)
      self._cpp('TS_ASSERT_LESS_THAN(sqrt(error/refNorm), {});'.format(epsMult*self._arch.epsilon))


  def tensor(self, node, resultName, maxValue = 512):
    ml = node.memoryLayout()
    size = ml.requiredReals()

    cpu_result_name = resultName
    cuda_result_name = "d_" + resultName

    spp = node.spp()
    isDense = spp.count_nonzero() == size
    if isDense:
      self.temporary(cpu_result_name, size)

      with self._cpp.For('int i = 0; i < {}; ++i'.format(size)):
        self._cpp('{}[i] = static_cast<{}>(i % {} + 1);'.format(resultName, self._arch.typename, maxValue))


    else:
      memory = ['0.0']*size
      nz = spp.nonzero()
      for entry in zip(*nz):
        addr = ml.address(entry)
        memory[addr] = str(float(addr % maxValue)+1.0)
      self.temporary(resultName, size, memory=memory)


    # copy data on GPU
    self.cuda_temporary(cuda_result_name, size)
    self._cpp(
      'cudaMemcpy({0}, {1}, sizeof({2}) * {3} \
      * yateto::tensor::num_elements_in_cluster, cudaMemcpyHostToDevice); CUDA_CHECK;'.format(
        cuda_result_name,
        cpu_result_name,
        self._arch.typename,
        size))
