import copy
import itertools
import re
import os
from yateto import Tensor
from .ast.cost import BoundingBoxCostEstimator
from .ast.node import Node
from .ast.visitor import ComputeOptimalFlopCount, FindIndexPermutations, FindTensors, FindPrefetchCapabilities
from .ast.transformer import *
from .codegen.cache import *
from .codegen.code import Cpp

from .codegen.visitor import *

from .controlflow.visitor import AST2ControlFlow
from .controlflow.transformer import *
from .gemm_configuration import GeneratorCollection, DefaultGeneratorCollection, BLASlike
from typing import List
from io import StringIO
from .controlflow.visitor import PrettyPrinter
from yateto.helper import GraphvisHelper


class Kernel(object):
  """The class represents a container which holds an abstract syntax tree as well as its
  corresponding name.

  An instance of a class allows the user to generate a primitive source code for a given tensor
  equation (i.e. in a give parse tree). The source code is generated based on nested loops.
  It won't achieve high performance but it is reliable and suitable for unit testing.
  """
  BASE_NAME = r'[a-zA-Z]\w*'
  VALID_NAME = r'^{}$'.format(BASE_NAME)

  def __init__(self, name, ast, prefetch=None):
    """
    Args:
      name (str): a kernel name
      ast (Union[Type[Node], List[Type[Node]]]): a root node of an abstract syntax tree
      prefetch (Union[Tensor, List[Tensor]]): a tensor or a list of tensors that have to be prefetched
                                              in a source code

    Raises:
      ValueError: if prefetch is neither a tensor nor a list of tensors
    """

    self.name = name

    # assure that ast is a list
    if isinstance(ast, list):
      self.ast = ast
    else:
      self.ast = [ast]


    # check whether prefetch follows specification and init it
    self._prefetch = None
    if prefetch is not None:

      if isinstance(prefetch, Tensor):
        self._prefetch = [prefetch]

      elif isinstance(prefetch, list) and all([isinstance(p, Tensor) for p in prefetch]):
        self._prefetch = prefetch

      else:
        raise ValueError('Prefetch must either be a Tensor (without indices) or a list of Tensors.')


    # Set a default value of cfg (stands for control flow graph)
    self.cfg = None

    # Initialize a non zero flop counter i.e. counter of useful floating point
    # operations because some matrices within a tensor equation can can be sparse
    self.nonZeroFlops = -1


  @classmethod
  def isValidName(cls, name):
    """Checks whether a given kernel name follows specification i.e. matches a regex

    Args:
      name (str): a kernel name

    Returns:
      bool: True if yes. Otherwise, False
    """
    return re.match(cls.VALID_NAME, name) is not None


  def prepareUntilUnitTest(self):

    # At this point self.ast is a list of kernels where each kernel is
    # an ast without annotation i.e. some nodes (Assign, Add, ScalarMultiplication, etc.)
    # don't have the target indices being set up.

    # Iterate through each ast and fix the above mentioned issue
    self.ast = [DeduceIndices().visit(ast) for ast in self.ast]

    # TODO: document
    ast2cf = AST2ControlFlow(simpleMemoryLayout=True)
    for ast in self.ast:
      ast2cf.visit(ast)

    self.cfg = ast2cf.cfg()
    self.cfg = LivenessAnalysis().visit(self.cfg)

  
  def prepareUntilCodeGen(self, costEstimator):
    """Analyzes an abstract syntax tree, generates and optimizes a control flow graph

    Args:
      costEstimator (Type[CostEstimator]): TODO
    """

    self.nonZeroFlops = 0
    for a in self.ast:
      ast = copy.deepcopy(a)
      ast = EquivalentSparsityPattern(groupSpp=False).visit(ast)
      ast = StrengthReduction(costEstimator).visit(ast)
      ast = SetSparsityPattern().visit(ast)
      self.nonZeroFlops += ComputeOptimalFlopCount().visit(ast)



    tmpASTs = list()
    prefetch = copy.copy(self._prefetch)

    # TODO: self.ast is a list but the list always contains one element.
    #  Questions: why do we need a list for this?
    for ast in self.ast:
      ast = EquivalentSparsityPattern().visit(ast)
      ast = StrengthReduction(costEstimator).visit(ast)
      ast = FindContractions().visit(ast)
      ast = ComputeMemoryLayout().visit(ast)
      permutationVariants = FindIndexPermutations().visit(ast)
      ast = SelectIndexPermutations(permutationVariants).visit(ast)
      ast = ImplementContractions().visit(ast)

      if self._prefetch is not None:
        prefetchCapabilities = FindPrefetchCapabilities().visit(ast)
        assignPf = AssignPrefetch(prefetchCapabilities, prefetch)
        ast = assignPf.visit(ast)
        prefetch = [pf for pf in prefetch if pf not in assignPf.assigned()]

      tmpASTs.append(ast)

    self.ast = tmpASTs


    # a nested helper function
    def print_cfd(cfg, optimization_name):
      to_print = True
      if to_print:
        marging = "="*80
        half_marging = "="*30
        print("{} {} {}".format(half_marging, optimization_name, half_marging))
        PrettyPrinter().visit(cfg)
        print("{}\n\n".format(marging))


    ast2cf = AST2ControlFlow()
    for ast in self.ast:
      ast2cf.visit(ast)

    self.cfg = ast2cf.cfg()
    #print_cfd(cfg=self.cfg, optimization_name="ast2ControlFlow")

    self.cfg = MergeScalarMultiplications().visit(self.cfg)
    #print_cfd(cfg=self.cfg, optimization_name="MergeScalarMultiplications")

    self.cfg = LivenessAnalysis().visit(self.cfg)
    #print_cfd(cfg=self.cfg, optimization_name="LivenessAnalysis")

    self.cfg = SubstituteForward().visit(self.cfg)
    #print_cfd(cfg=self.cfg, optimization_name="SubstituteForward")

    self.cfg = SubstituteBackward().visit(self.cfg)
    #print_cfd(cfg=self.cfg, optimization_name="SubstituteBackward")

    self.cfg = RemoveEmptyStatements().visit(self.cfg)
    #print_cfd(cfg=self.cfg, optimization_name="RemoveEmptyStatements")

    self.cfg = MergeActions().visit(self.cfg)
    print_cfd(cfg=self.cfg, optimization_name="MergeActions")

    print('='*80)

    
class KernelFamily(object):
  """ The class represents a table of similar kernels. A kernel family has a stric rule w.r.t.
  kernel names i.e. a base kernel name followed by a group index within parantethis.
  It allows to use group indices as keys within the table and values as concrete abstract syntax
  trees.

  An instance of a class can generate a simple source code for all tensor equations involved
  in a table. It is down by traversing the table and delegating the source code generation to
  each kernel instance.
  """
  GROUP_INDEX = r'\((0|[1-9]\d*)\)'
  VALID_NAME = r'^{}({})$'.format(Kernel.BASE_NAME, GROUP_INDEX)

  def __init__(self):
    self._kernels = dict()  # Dict[int, Type[Node]]
    self.name = None  # str
    self._stride = None  # Tuple[int]
  
  def items(self):
    return self._kernels.items()
  
  def __len__(self):
    return max(self._kernels.keys()) + 1


  @classmethod  
  def baseName(self, name):
    return re.match(Kernel.BASE_NAME, name).group(0)


  @classmethod
  def isValidName(cls, name):
    """Checks whether a given name follows a class specification i.e regex cls.VALID_NAME

    Args:
      name (str): a kernel name

    Returns:
      bool: True, if yes. Otherwise, False
    """
    return re.match(cls.VALID_NAME, name) is not None


  @classmethod
  def group(cls, name):
    """Extracts a group index (which is inside of parentheses)

    Args:
      name (str): a kernel name

    Returns:
      int: a group index

    Examples:
      >>> name = 'aTensor(4)'
      >>> KernelFamily.group(name)
      4

    """
    m = re.search(cls.GROUP_INDEX, name)
    return int(m.group(1))


  def setStride(self, stride):
    """
    Args:
      stride (Tuple[int]):
    """
    self._stride = stride


  def stride(self):
    """
    Returns:
      Tuple[int]: distances of each dimension form zero element
    """
    if self._stride is not None:
      return self._stride
    return (1,)


  @classmethod
  def linear(cls, stride, group):
    """Generates an index (an integer value) from an index set (an element from an index(iteration
    space)) and strides of each dimension

    Args:
      stride (Tuple[int]): a distance of each dimension from the first element i.e. zero element
      group (Tuple[int]): a set of indices

    Returns:
      int: a linear index
    """

    assert len(stride) == len(group)

    index = 0
    for i, p in enumerate(group):
      index += p * stride[i]
    return index


  def add(self, name, ast, prefetch=None):
    """

    Args:
      name (str): a kernel name
      ast (Type[Node]): a root node of an abstract syntax tree
      prefetch (Union[Tensor, List[Tensor], None]): specifies tensors for which yateto will generate
                                                    data prefetch with a source code
    Returns:

    """
    baseName = self.baseName(name)
    if not self.name:
      self.name = baseName

    assert baseName == self.name

    # extract a group index from a kernel name
    group = self.group(name)

    internalName = '_{}_{}'.format(baseName, group)
    self._kernels[group] = Kernel(internalName, ast, prefetch)


  def kernels(self):
    """
    Returns:
      List[Kernel]: all kernels inside of a table
    """
    return self._kernels.values()


  def prepareUntilUnitTest(self):
    for kernel in self._kernels.values():
      kernel.prepareUntilUnitTest()
  
  def prepareUntilCodeGen(self, costEstimator):
    for kernel in self._kernels.values():
      kernel.prepareUntilCodeGen(costEstimator)



def simpleParameterSpace(*args):
  """Generates all possible combinations of indices

  The function takes a bunch of integers as an arbitrary list. Each element represents a size of
  the corresponding dimension. For example: simpleParameterSpace(2,3,4) means that there are
  3 dimensions. The first one spans between 0 and 1, the second - between 0 and 2, the third one -
  between 0 and 4. The function returns all possible combinations of indices called index or
  iteration space.

  Args:
    *args (int): an arbitrary list of integers

  Returns:
    List[Tuple[int]]: a cross product of a given dimensions

  Examples:
    >>> simpleParameterSpace(1,2,3)
    [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0), (0, 1, 1), (0, 1, 2)]
    >>> simpleParameterSpace(4,2)
    [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)]

  """
  return list(itertools.product(*[list(range(i)) for i in args]))


def parameterSpaceFromRanges(*args):
  """Generates all possible combinations of indices

  The function takes a bunch of integers as an arbitrary list. Each element represents a size of
  the corresponding dimension. For example: simpleParameterSpace(2,3,4) means that there are
  3 dimensions. The first one spans between 0 and 1, the second - between 0 and 2, the third one -
  between 0 and 4. The function returns all possible combinations of indices called index or
  iteration space.

  Args:
    *args (int): an arbitrary list of integers

  Returns:
    List[Tuple[int]]: a cross product of a given dimensions

  Examples:
    >>> simpleParameterSpace(1,2,3)
    [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0), (0, 1, 1), (0, 1, 2)]
    >>> simpleParameterSpace(4,2)
    [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)]

  """
  return list(itertools.product(*[list(i) for i in args]))


class Generator(object):
  INIT_FILE_NAME = 'init'
  TENSORS_FILE_NAME = 'tensor'
  KERNELS_FILE_NAME = 'kernel'
  ROUTINES_FILE_NAME = 'subroutine'
  UNIT_TESTS_FILE_NAME = 'KernelTest.t'
  HEADER_GUARD_SUFFIX = 'H_'
  SUPPORT_LIBRARY_HEADER = 'yateto.h'
  TEST_CLASS = 'KernelTestSuite'
  TEST_NAMESPACE = 'unit_test'
  
  class FileNames(object):
    HEADER = 'h'
    CPP = 'cpp'

    def __init__(self, outputDir, name):
      self.hName = '{}.{}'.format(name, self.HEADER)
      self.cppName = '{}.{}'.format(name, self.CPP)
      self.h = os.path.join(outputDir, self.hName)
      self.cpp = os.path.join(outputDir, self.cppName)
  
  def __init__(self, arch):
    """
    arch (Architecture): an instance of Architecture class which holds basic information
                         about a target compute architecture
    """
    self._kernels = list()
    self._kernelFamilies = dict()
    self._arch = arch

  def arch(self):
    return self._arch
  
  def add(self, name: str, ast: Node, prefetch=None):
    """Adds a give abstract syntax tree to either a table of Kernel Families or to a list of Kernels

    A decision whether a kernel belongs to a Family or not is based on a kernel name. A kernel
    belongs to a family if a kernel name contains parentheses

    Args:
      name (str): a name of a kernel (a tensor expression)
      ast (Type[Node]): a root node of a tensor expression (abstract syntax tree)
      prefetch (Union[Tensor, List[Tensor], None]): specifies which tensors have to be prefetched in
                                                    a source code

    Raises:
      ValueError: if a give kernel name violates a yateto kernel name convention.
                  See, class Kernel for details
    """

    if KernelFamily.isValidName(name):
      baseName = KernelFamily.baseName(name)
      if baseName not in self._kernelFamilies:
        self._kernelFamilies[baseName] = KernelFamily()
      self._kernelFamilies[baseName].add(name, ast, prefetch)
    else:      
      if not Kernel.isValidName(name):
        raise ValueError('Kernel name invalid (must match regexp {}): {}'.format(Kernel.VALID_NAME, name))
      kernel = Kernel(name, ast, prefetch)
      self._kernels.append(kernel)

  def kernels(self):
    return [kernel for kernel in self._kernels] + [kernel for family in self._kernelFamilies.values() for kernel in family.kernels()]


  def addFamily(self, name: str, parameterSpace, astGenerator, prefetchGenerator=None):
    """Generates a kernel family from a given parameterized abstract syntrax tree generator.

    The function assumes that parameter space looks like a multidimensional tensor of integers.
    Each kernel corresponds ot a particular set of indices (a tensor element). To name each kernel,
    the function linearises each set in order to get an integer out of it.

    Args:
      name (str): a name of a kernel family
      parameterSpace (List[Tuple[int]]): sets of possible indices i.e. index(iteration) space

      astGenerator (Callable[[List[int]], []]): a callback to a parameterized abstract
                                                syntrax tree

      prefetchGenerator (Callable[[List[int]], [Tensor, List[Tensors]]]): a callback to a
                                                     parameterized prefetch generator.
                                                     The callback a set of indices and return
                                                     either a tensor, or a list if tensors that
                                                     have to be prefetched during in a source
                                                     code
    """

    # Create a new entry in a table of kernel families if it has not been allocated before
    if name not in self._kernelFamilies:
      self._kernelFamilies[name] = KernelFamily()

    family = self._kernelFamilies[name]


    # get the maximum indices from a parameter space
    pmax = max(parameterSpace)

    stride = [1]

    for i in range(len(pmax) - 1):
      stride.append(stride[i] * (pmax[i] + 1))


    stride = tuple(stride)

    family.setStride(stride)
    for p in parameterSpace:
      indexedName = '{}({})'.format(name, KernelFamily.linear(stride, p))
      ast = astGenerator(*p)
      prefetch = prefetchGenerator(*p) if prefetchGenerator is not None else None
      family.add(indexedName, ast, prefetch)


  def _headerGuardName(self, namespace, fileBaseName):
    partlist = namespace.upper().split('::') + [fileBaseName.upper(), self.HEADER_GUARD_SUFFIX]
    return '_'.join(partlist)


  def generate( self,
                outputDir: str,
                namespace='yateto',
                gemm_cfg: GeneratorCollection=None,
                costEstimator=BoundingBoxCostEstimator):


    if not gemm_cfg:
      gemm_cfg = DefaultGeneratorCollection(self._arch)


    print('Deducing indices...')
    for kernel in self._kernels:
      kernel.prepareUntilUnitTest()
    for family in self._kernelFamilies.values():
      family.prepareUntilUnitTest()


    # allocate file instances for gode generation
    fUT = self.FileNames(outputDir, self.UNIT_TESTS_FILE_NAME)
    fKernels = self.FileNames(outputDir, self.KERNELS_FILE_NAME)
    fRoutines = self.FileNames(outputDir, self.ROUTINES_FILE_NAME)
    fTensors = self.FileNames(outputDir, self.TENSORS_FILE_NAME)
    fInit = self.FileNames(outputDir, self.INIT_FILE_NAME)


    print('Generating unit tests...')
    with Cpp(fUT.h) as cpp:
      with cpp.HeaderGuard(self._headerGuardName(namespace, self.UNIT_TESTS_FILE_NAME.replace('.', '_'))):
        cpp.includeSys('cxxtest/TestSuite.h')
        cpp.include(fKernels.hName)
        cpp.include(fInit.hName)
        with cpp.PPIfndef('NDEBUG'):
          cpp('long long libxsmm_num_total_flops = 0;')
          cpp('long long pspamm_num_total_flops = 0;')
        with cpp.Namespace(namespace):
          with cpp.Namespace(self.TEST_NAMESPACE):
            cpp.classDeclaration(self.TEST_CLASS)
        with cpp.Class('{}::{}::{} : public CxxTest::TestSuite'.format(namespace, self.TEST_NAMESPACE, self.TEST_CLASS)):
          cpp.label('public')

          for kernel in self._kernels:
            UnitTestGenerator(self._arch).generate(cpp, kernel.name, kernel.name, kernel.cfg, gemm_cfg)

          for family in self._kernelFamilies.values():
            for group, kernel in family.items():
              UnitTestGenerator(self._arch).generate(cpp, kernel.name, family.name, kernel.cfg, gemm_cfg, group)


    print('Optimizing ASTs...')
    for kernel in self._kernels:
      print(kernel.name)
      kernel.prepareUntilCodeGen(costEstimator)
    for family in self._kernelFamilies.values():
      print(family.name)
      family.prepareUntilCodeGen(costEstimator)


    print('Generating kernels...')
    cache = RoutineCache()
    optKernelGenerator = OptimisedKernelGenerator(self._arch, cache)


    kernelSource = StringIO()
    kernelSourceContent = ''
    with Cpp(kernelSource) as cpp:      
      cpp.includeSys('cassert')
      cpp.includeSys('cstring')
      cpp.includeSys('cstdlib')
      cpp.include(fRoutines.hName)
      with Cpp(fKernels.h) as header:
        with header.HeaderGuard(self._headerGuardName(namespace, self.KERNELS_FILE_NAME)):
          header.includeSys('cmath')
          header.includeSys('limits')
          header.include(fTensors.hName)
          cpp.include(fKernels.hName)
          with cpp.Namespace(namespace):
            with header.Namespace(namespace):
              for kernel in self._kernels:
                kernelOutline = optKernelGenerator.generateKernelOutline(kernel.nonZeroFlops, kernel.cfg, gemm_cfg)
                optKernelGenerator.generate(cpp, header, kernel.name, [kernelOutline])
              for family in self._kernelFamilies.values():
                kernelOutlines = [None] * len(family)
                for group, kernel in family.items():
                  kernelOutlines[group] = optKernelGenerator.generateKernelOutline(kernel.nonZeroFlops, kernel.cfg, gemm_cfg)
                optKernelGenerator.generate(cpp, header, family.name, kernelOutlines, family.stride())
      kernelSourceContent = kernelSource.getvalue()


    with Cpp(fKernels.cpp) as cpp:
      for gemm_tool in gemm_cfg.selected:
        for inc in gemm_tool.includes:
          cpp.include(inc)
        if isinstance(gemm_tool, BLASlike):
          cpp(gemm_tool.c_code_init)
      cpp.out.write(kernelSourceContent)      


    print('Calling external code generators...')
    with Cpp(fRoutines.h) as header:
      with header.HeaderGuard(self._headerGuardName(namespace, self.ROUTINES_FILE_NAME)):
        cache.generate(header, fRoutines.cpp)
    
    tensors = dict()
    for kernel in self._kernels:
      tensors.update(FindTensors().visit(kernel.ast))
    for family in self._kernelFamilies.values():
      for group, kernel in family.items():
        tensors.update(FindTensors().visit(kernel.ast))

    print('Generating initialization code...')
    #initGen = CudaInitializerGenerator(self._arch, sorted(tensors.values(), key=lambda x: x.name()))
    initGen = InitializerGenerator(self._arch, sorted(tensors.values(), key=lambda x: x.name()))
    with Cpp(fTensors.h) as header:
      with header.HeaderGuard(self._headerGuardName(namespace, self.TENSORS_FILE_NAME)):
        with header.Namespace(namespace):
          initGen.generateTensorsH(header)


    with Cpp(fTensors.cpp) as cpp:
      cpp.include(fTensors.hName)
      with cpp.Namespace(namespace):
        initGen.generateTensorsCpp(cpp)


    with Cpp(fInit.h) as header:
      with header.HeaderGuard(self._headerGuardName(namespace, self.INIT_FILE_NAME)):
        header.include(fTensors.hName)
        header.include(self.SUPPORT_LIBRARY_HEADER)
        with header.Namespace(namespace):
          initGen.generateInitH(header)


    with Cpp(fInit.cpp) as cpp:
      cpp.include(fInit.hName)
      with cpp.Namespace(namespace):
        initGen.generateInitCpp(cpp)


####################################################################################################
#                                           CUDA
####################################################################################################
from .codegen.cuda_visitor import CudaKernelGenerator, CudaOptimisedKernelGenerator
from .codegen.cuda_visitor import CudaUnitTestGenerator, CudaInitializerGenerator

class CudaGenerator(object):
  INIT_FILE_NAME = 'init'
  TENSORS_FILE_NAME = 'tensor'
  KERNELS_FILE_NAME = 'kernel'
  ROUTINES_FILE_NAME = 'subroutine'
  UNIT_TESTS_FILE_NAME = 'KernelTest.t'
  HEADER_GUARD_SUFFIX = 'H_'
  SUPPORT_LIBRARY_HEADER = 'yateto.h'
  TEST_CLASS = 'KernelTestSuite'
  TEST_NAMESPACE = 'unit_test'


  class FileNames(object):
    HEADER = 'h'
    CPP = 'cpp'

    def __init__(self, outputDir, name):
      # names of header, source, cuda_source files
      self.hName = '{}.{}'.format(name, self.HEADER)
      self.cppName = '{}.{}'.format(name, self.CPP)

      # paths to the source files
      self.h = os.path.join(outputDir, self.hName)
      self.cpp = os.path.join(outputDir, self.cppName)


  def __init__(self, arch):
    """
    arch (Architecture): an instance of Architecture class which holds basic information
                         about a target compute architecture
    """
    self._kernels = list() # List[Type[Kernel]]
    self._kernelFamilies = dict()  # Dict[str, List[Type[KernelFamily]]]
    self._arch = arch


  def arch(self):
    """
    Returns:
      Architecture: a reference to a specific Architecture instance which was given to a Generator

    """
    return self._arch


  def add(self, name: str, ast: Node, prefetch=None):
    """Adds a give abstract syntax tree to either a table of Kernel Families or to a list of Kernels

    A decision whether a kernel belongs to a Family or not is based on a kernel name. A kernel
    belongs to a family if a kernel name contains parentheses

    Args:
      name (str): a name of a kernel (a tensor expression)
      ast (Type[Node]): a root node of a tensor expression (abstract syntax tree)
      prefetch (Union[Tensor, List[Tensor], None]): specifies which tensors have to be prefetched in
                                                    a source code

    Raises:
      ValueError: if a give kernel name violates a yateto kernel name convention.
                  See, class Kernel for details
    """

    if KernelFamily.isValidName(name):
      baseName = KernelFamily.baseName(name)

      if baseName not in self._kernelFamilies:
        self._kernelFamilies[baseName] = KernelFamily()

      self._kernelFamilies[baseName].add(name, ast, prefetch)

    else:
      if not Kernel.isValidName(name):
        raise ValueError('Kernel name invalid (must match regexp {}): {}'.format(Kernel.VALID_NAME,
                                                                                 name))
      kernel = Kernel(name, ast, prefetch)
      self._kernels.append(kernel)


  def kernels(self):
    """
    Returns:
      List[Type[Kernel]]: all kernels which have been added to a generator
    """
    return [kernel for kernel in self._kernels] + [kernel for family in
                                                   self._kernelFamilies.values() for kernel in
                                                   family.kernels()]

  def addFamily(self, name: str, parameterSpace, astGenerator, prefetchGenerator=None):
    """Generates a kernel family from a given parameterized abstract syntrax tree generator.

    The function assumes that parameter space looks like a multidimensional tensor of integers.
    Each kernel corresponds ot a particular set of indices (a tensor element). To name each kernel,
    the function linearises each set in order to get an integer out of it.

    Args:
      name (str): a name of a kernel family
      parameterSpace (List[Tuple[int]]): sets of possible indices i.e. index(iteration) space

      astGenerator (Callable[[List[int]], []]): a callback to a parameterized abstract
                                                syntrax tree

      prefetchGenerator (Callable[[List[int]], [Tensor, List[Tensors]]]): a callback to a
                                                     parameterized prefetch generator.
                                                     The callback a set of indices and return
                                                     either a tensor, or a list if tensors that
                                                     have to be prefetched during in a source
                                                     code
    """

    if name not in self._kernelFamilies:
      self._kernelFamilies[name] = KernelFamily()

    family = self._kernelFamilies[name]

    # get the larges indices within a give parameter space
    max_parameter = max(parameterSpace)

    # compute a linear distance of each dimension from first element
    # of the index (iteration) space i.e. from element zero
    stride = [1]
    for i in range(len(max_parameter) - 1):
      stride.append(stride[i] * (max_parameter[i] + 1))

    stride = tuple(stride)
    family.setStride(stride)

    # iterate through all possible sets of indices and generate a kernel for each of them
    for indices in parameterSpace:

      # compute a name for a kernel
      # NOTE: the name is a combination of the base name as well as stride and group indices
      indexedName = '{}({})'.format(name, KernelFamily.linear(stride=stride, group=indices))

      # generate a concrete instance of an abstracr syntax tree from an ast generator
      ast = astGenerator(*indices)

      # extract
      prefetch = prefetchGenerator(*indices) if prefetchGenerator is not None else None

      # add a kernel to a particular kernel family
      family.add(indexedName, ast, prefetch)


  def _headerGuardName(self, namespace, fileBaseName):
    partlist = namespace.upper().split('::') + [fileBaseName.upper(), self.HEADER_GUARD_SUFFIX]
    return '_'.join(partlist)


  def generate(self,
               outputDir: str,
               namespace='yateto',
               gemm_cfg: GeneratorCollection = None,
               costEstimator=BoundingBoxCostEstimator):
    """Generates a source code for given kernels i.e. for given abstract syntax trees

    Args:
      outputDir (str): a path to the source code output directory
      namespace (str): a name space which yateto will use for generated code
      gemm_cfg (GeneratorCollection): a set of source code generators (for matrix-matrix
                                      multiplications)
      costEstimator (Type[CostEstimator]): TODO
    """

    if not gemm_cfg:
      gemm_cfg = DefaultGeneratorCollection(self._arch)

    print('Deducing indices...')
    for kernel in self._kernels:
      kernel.prepareUntilUnitTest()

    for family in self._kernelFamilies.values():
      family.prepareUntilUnitTest()

    # define file names for source code generation
    fUT = self.FileNames(outputDir, self.UNIT_TESTS_FILE_NAME)
    fKernels = self.FileNames(outputDir, self.KERNELS_FILE_NAME)
    fRoutines = self.FileNames(outputDir, self.ROUTINES_FILE_NAME)
    fTensors = self.FileNames(outputDir, self.TENSORS_FILE_NAME)
    fInit = self.FileNames(outputDir, self.INIT_FILE_NAME)

    ################################################################################################
    print('Generating unit tests...')
    full_test_class_name = "{}::{}::{}".format(namespace, self.TEST_NAMESPACE, self.TEST_CLASS)
    with Cpp(fUT.h) as unit_test_header:

      header_guards_name = self._headerGuardName(namespace,
                                           self.UNIT_TESTS_FILE_NAME.replace('.', '_'))
      with unit_test_header.HeaderGuard(header_guards_name):

        # write down all necessary include files
        unit_test_header.includeSys('cxxtest/TestSuite.h')
        unit_test_header.include(fKernels.hName)
        unit_test_header.include(fInit.hName)

        '''
        with unit_test_header.PPIfndef('NDEBUG'):
          # define profiling counters
          unit_test_header('long long libxsmm_num_total_flops = 0;')
          unit_test_header('long long pspamm_num_total_flops = 0;')
        '''


        with unit_test_header.Namespace(namespace):
          with unit_test_header.Namespace(self.TEST_NAMESPACE):
            unit_test_header.classDeclaration(self.TEST_CLASS)


        # declare the test suite class in the header file
        with unit_test_header.Class('{} : public CxxTest::TestSuite'.format(full_test_class_name)):
          unit_test_header.label('public')


          # iterate through all kernels and declare then inside of the header file
          for kernel in self._kernels:
            test_function_name = CudaUnitTestGenerator.CXXTEST_PREFIX + kernel.name
            unit_test_header("void {}();".format(test_function_name))


          for family in self._kernelFamilies.values():
            for group, kernel in family.items():
              test_function_name = CudaUnitTestGenerator.CXXTEST_PREFIX + kernel.name
              unit_test_header("void {}();".format(test_function_name))


    # generate the source files for the kernel test suite
    with Cpp(fUT.cpp) as unit_test_source:
      # specify all necessary hedear files which the unit tests depend on
      unit_test_source.includeSys('cxxtest/TestSuite.h')
      unit_test_source.include(fUT.hName)
      unit_test_source.include(fInit.hName)
      unit_test_source.include(fKernels.hName)
      unit_test_source.include(fTensors.hName)
      unit_test_source.include("device_utils.h")
      unit_test_source.include('Util.h')
      unit_test_source.include('TensorView.h')


      # generate the source code from parse trees for unit tests
      for kernel in self._kernels:
        CudaUnitTestGenerator(self._arch).generate(cpp=unit_test_source,
                                                   testName=kernel.name,
                                                   kernelClass=kernel.name,
                                                   cfg=kernel.cfg,
                                                   gemm_cfg=gemm_cfg,
                                                   function_namespace=full_test_class_name)

      for family in self._kernelFamilies.values():
        for group, kernel in family.items():
          CudaUnitTestGenerator(self._arch).generate(cpp=unit_test_source,
                                                     testName=kernel.name,
                                                     kernelClass=family.name,
                                                     cfg=kernel.cfg,
                                                     gemm_cfg=gemm_cfg,
                                                     index=group,
                                                     function_namespace=full_test_class_name)

    ################################################################################################
    print('Optimizing ASTs...')
    for kernel in self._kernels:
      print(kernel.name)
      kernel.prepareUntilCodeGen(costEstimator)

    for family in self._kernelFamilies.values():
      print(family.name)
      family.prepareUntilCodeGen(costEstimator)


    #TODO: the following block is needed for rendering parse trees
    debug = False
    if debug:
      # render and save all optimized parse trees in image files
      parse_tree_visuzlizer = GraphvisHelper(output_dir='./parse-tree-optimized')
      for kernel in self.kernels():
        parse_tree_visuzlizer.visualize(tree_name=kernel.name,
                                        tree_root=kernel.ast[0],
                                        is_display=False)


    ################################################################################################
    print('Generating kernels...')
    cache = RoutineCache()
    optKernelGenerator = CudaOptimisedKernelGenerator(self._arch, cache)

    kernelSource = StringIO()
    kernelSourceContent = ''
    with Cpp(kernelSource) as cpp:
      cpp.includeSys('cassert')
      cpp.includeSys('cstring')
      cpp.includeSys('cstdlib')
      cpp.include(fRoutines.hName)

      with Cpp(fKernels.h) as header:
        with header.HeaderGuard(self._headerGuardName(namespace, self.KERNELS_FILE_NAME)):
          header.includeSys('cmath')
          header.includeSys('limits')
          header.include(fTensors.hName)
          cpp.include(fKernels.hName)
          with cpp.Namespace(namespace):
            with header.Namespace(namespace):

              for kernel in self._kernels:
                kernelOutline = optKernelGenerator.generateKernelOutline(kernel.nonZeroFlops,
                                                                         kernel.cfg,
                                                                         gemm_cfg)

                # generate both code body for both source and header files
                optKernelGenerator.generate(cpp, header, kernel.name, [kernelOutline])

              for family in self._kernelFamilies.values():
                kernelOutlines = [None] * len(family)

                for group, kernel in family.items():
                  kernelOutlines[group] = optKernelGenerator.generateKernelOutline(kernel.nonZeroFlops,
                                                                                   kernel.cfg,
                                                                                   gemm_cfg)

                optKernelGenerator.generate(cpp, header, family.name, kernelOutlines,
                                            family.stride())


      kernelSourceContent = kernelSource.getvalue()

    with Cpp(fKernels.cpp) as cpp:
      for gemm_tool in gemm_cfg.selected:

        for inc in gemm_tool.includes:
          cpp.include(inc)

        if isinstance(gemm_tool, BLASlike):
          cpp(gemm_tool.c_code_init)

      # add the kerlnel body generated above to a text source file
      cpp.out.write(kernelSourceContent)


    ################################################################################################
    print('Calling external code generators...')
    with Cpp(fRoutines.h) as header:
      with header.HeaderGuard(self._headerGuardName(namespace, self.ROUTINES_FILE_NAME)):
        cache.generate(header, fRoutines.cpp)


    tensors = dict()
    for kernel in self._kernels:
      tensors.update(FindTensors().visit(kernel.ast))

    for family in self._kernelFamilies.values():
      for group, kernel in family.items():
        tensors.update(FindTensors().visit(kernel.ast))


    ################################################################################################
    print('Generating initialization code...')
    initGen = CudaInitializerGenerator(arch=self._arch,
                                       tensors=sorted(tensors.values(), key=lambda x: x.name()))

    with Cpp(fTensors.h) as header:
      with header.HeaderGuard(self._headerGuardName(namespace, self.TENSORS_FILE_NAME)):
        with header.Namespace(namespace):
          initGen.generateTensorsH(header)


    with Cpp(fTensors.cpp) as cpp:
      cpp.include(fTensors.hName)
      with cpp.Namespace(namespace):
        initGen.generateTensorsCpp(cpp)


    with Cpp(fInit.h) as header:
      with header.HeaderGuard(self._headerGuardName(namespace, self.INIT_FILE_NAME)):
        header.include(fTensors.hName)
        header.include(self.SUPPORT_LIBRARY_HEADER)
        with header.Namespace(namespace):
          initGen.generateInitH(header)


    with Cpp(fInit.cpp) as cpp:
      cpp.include(fInit.hName)
      with cpp.Namespace(namespace):
        initGen.generateInitCpp(cpp)
