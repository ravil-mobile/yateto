import string
from ..ast.indices import Indices, Range
from ..ast.node import IndexedTensor
from ..memory import DenseMemoryLayout
from .common import forLoops, TensorDescription, IndexedTensorDescription
from . import copyscaleadd, indexsum, log, product

class KernelFactory(object):
  ERROR_NAME = '_error'

  def __init__(self, cpp, arch):
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

  def temporary(self, bufname, size, iniZero=False, memory=list()):
    assert(iniZero == False or len(memory) == 0)
    if self._arch.onHeap(size):
      if memory:
        raise NotImplementedError('Direct temporary initialization is not supported for heap-allocated memory.')
      if len(self._freeList) == 0:
        self._cpp('int {};'.format(self.ERROR_NAME))
      self._cpp('{}* {};'.format(self._arch.typename, bufname))
      self._cpp('{} = posix_memalign(reinterpret_cast<void**>(&{}), {}, {}*sizeof({}));'.format(
                  self.ERROR_NAME,
                  bufname,
                  self._arch.alignment,
                  size,
                  self._arch.typename))
      if iniZero:
        self._cpp.memset(bufname, size, self._arch.typename)
      self._freeList.append(bufname)
    else:
      ini = ''
      if iniZero:
        ini = ' = {}'
      elif memory:
        ini = ' = {{{}}}'.format(', '.join(memory))
      self._cpp('{} {}[{}] __attribute__((aligned({}))){};'.format(self._arch.typename, bufname, size, self._arch.alignment, ini))


  def freeTmp(self):
    for free in self._freeList:
      self._cpp('free({});'.format(free))
    self._freeList = []


  def _indices(self, var):
    """Renames indices of a tensor in alphabetical order

    NOTE: renamed indices are used for local computation within a generated source code

    Args:
      var (Variable): a unit of an execution block (usually a tensor)

    Returns:
      Indices: renamed tensor indices

    Examples:
      >>> from yateto.memory import DenseMemoryLayout
      >>> tensor_shape = (4,3)
      >>> layout = DenseMemoryLayout(shape=tensor_shape)
      >>> from yateto.controlflow.graph import Variable
      >>> variable = Variable(name='Tensor', writable=True, memoryLayout=layout)
      >>> from yateto.arch import Architecture
      >>> arch = Architecture(name='hsw', precision='D', alignment=32)
      >>> from io import StringIO
      >>> from yateto.codegen.factory import KernelFactory
      >>> factory = KernelFactory(cpp=StringIO(), arch=arch)
      >>> factory._indices(variable)
      (a=4,b=3)

    """
    shape = var.memoryLayout().shape()
    return Indices(string.ascii_lowercase[:len(shape)], shape)


class OptimisedKernelFactory(KernelFactory):
  def __init__(self, cpp, arch):
    super().__init__(cpp, arch)

  def create_LoopOverGEMM(self, node, result, arguments, add, scalar, prefetchName, routineCache, gemm_cfg):
    assert len(arguments) == 2
    description = log.Description(
      alpha = scalar,
      add = add,
      result = IndexedTensorDescription.fromNode(result, node),
      leftTerm = IndexedTensorDescription.fromNode(arguments[0], node.leftTerm()),
      rightTerm = IndexedTensorDescription.fromNode(arguments[1], node.rightTerm()),
      loopIndices = node.loopIndices(),
      transA = node.transA(),
      transB = node.transB(),
      prefetchName = prefetchName
    )
    generator = log.generator(self._arch, description)
    return generator.generate(self._cpp, routineCache, gemm_cfg)
  
  def create_IndexSum(self, node, result, arguments, add, scalar, prefetchName, routineCache, gemm_cfg):
    assert len(arguments) == 1
    description = indexsum.Description(
      alpha = scalar,
      add = add,
      result = IndexedTensorDescription.fromNode(result, node),
      term = IndexedTensorDescription.fromNode(arguments[0], node.term())
    )
    generator = indexsum.generator(self._arch, description)
    return generator.generate(self._cpp, routineCache)
  
  def create_Product(self, node, result, arguments, add, scalar, prefetchName, routineCache, gemm_cfg):
    assert len(arguments) == 2
    description = product.Description(
      alpha = scalar,
      add = add,
      result = IndexedTensorDescription.fromNode(result, node),
      leftTerm = IndexedTensorDescription.fromNode(arguments[0], node.leftTerm()),
      rightTerm = IndexedTensorDescription.fromNode(arguments[1], node.rightTerm())
    )
    generator = product.generator(self._arch, description)
    return generator.generate(self._cpp, routineCache)


  def simple(self, result, term, add, scalar, routineCache):
    """Prepares data to generate a tensor equation in a form: B = beta * B + alpha * A

    Args:
      result (Variable):
      term (Variable):
      add (bool): enables or disables the first term of the lhs
      scalar (Union[Scalar, float]):
      routineCache:

    Returns:
      int: a theoretical number of floating point operation required for a generated part of code
    """

    # prepare descriptions of each term
    beta = 1.0 if add else 0.0
    rhs_description = IndexedTensorDescription(name=str(result),
                                               indices=self._indices(result),
                                               memoryLayout=result.memoryLayout(),
                                               eqspp=result.eqspp())

    lhs_description = IndexedTensorDescription(name=str(term),
                                               indices=self._indices(term),
                                               memoryLayout=term.memoryLayout(),
                                               eqspp=term.eqspp())

    # prepare a description of a tensor operation
    description = copyscaleadd.Description(alpha=scalar,
                                           beta=beta,
                                           result=rhs_description,
                                           term=lhs_description)

    # init a code generation subroutine
    generator = copyscaleadd.generator(self._arch, description)

    # generate a piece of the source code
    # and count the number of floating point operations
    # for this part of the code
    return generator.generate(self._cpp, routineCache)


class UnitTestFactory(KernelFactory):
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
    """Prepares data to generate a tensor equation in a form: B = beta * B + alpha * A, for unit
    testing

    TODO: Check the description above

    NOTE: some data of the right hand side can be partially used for computations
    on the hand left side

    Args:
      result (Variable): right hand side of an equation
      term (Variable): left habd side of an equation
      add (bool): enables or disables the first term of the lhs
      scalar (Union[Scalar, float, None]): TODO
      routineCache (Union[None, RoutineCache]): TODO

    Returns:
      int: a number of floating point operations
    """
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
        self._cpp( 'double ref = {};'.format(refTerm) )
        self._cpp( 'double diff = ref - {};'.format(targetTerm) )
        self._cpp( 'error += diff * diff;' )
        self._cpp( 'refNorm += ref * ref;' )
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

    spp = node.spp()
    isDense = spp.count_nonzero() == size
    if isDense:
      self.temporary(resultName, size)
      with self._cpp.For('int i = 0; i < {}; ++i'.format(size)):
        self._cpp('{}[i] = static_cast<{}>(i % {} + 1);'.format(resultName, self._arch.typename, maxValue))
    else:
      memory = ['0.0']*size
      nz = spp.nonzero()
      for entry in zip(*nz):
        addr = ml.address(entry)
        memory[addr] = str(float(addr % maxValue)+1.0)
      self.temporary(resultName, size, memory=memory)


