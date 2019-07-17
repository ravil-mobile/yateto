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


class Kernel(object):
  BASE_NAME = r'[a-zA-Z]\w*'
  VALID_NAME = r'^{}$'.format(BASE_NAME)

  def __init__(self, name, ast, prefetch=None):
    self.name = name
    if isinstance(ast, list):
      self.ast = ast
    else:
      self.ast = [ast]
    self._prefetch = None
    if prefetch is not None:
      if isinstance(prefetch, Tensor):
        self._prefetch = [prefetch]
      elif isinstance(prefetch, list) and all([isinstance(p, Tensor) for p in prefetch]):
        self._prefetch = prefetch
      else:
        raise ValueError('Prefetch must either be a Tensor (without indices) or a list of Tensors.')
    self.cfg = None
    self.nonZeroFlops = -1

  @classmethod
  def isValidName(cls, name):
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

    self.nonZeroFlops = 0
    for a in self.ast:
      ast = copy.deepcopy(a)
      ast = EquivalentSparsityPattern(groupSpp=False).visit(ast)
      ast = StrengthReduction(costEstimator).visit(ast)
      ast = SetSparsityPattern().visit(ast)
      self.nonZeroFlops += ComputeOptimalFlopCount().visit(ast)

    tmpASTs = list()
    prefetch = copy.copy(self._prefetch)
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


    def print_cfd(cfg, optimization_name):
      to_print = False
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
    print_cfd(cfg=self.cfg, optimization_name="ast2ControlFlow")

    self.cfg = MergeScalarMultiplications().visit(self.cfg)
    print_cfd(cfg=self.cfg, optimization_name="MergeScalarMultiplications")

    self.cfg = LivenessAnalysis().visit(self.cfg)
    print_cfd(cfg=self.cfg, optimization_name="LivenessAnalysis")

    self.cfg = SubstituteForward().visit(self.cfg)
    print_cfd(cfg=self.cfg, optimization_name="SubstituteForward")

    self.cfg = SubstituteBackward().visit(self.cfg)
    print_cfd(cfg=self.cfg, optimization_name="SubstituteBackward")

    self.cfg = RemoveEmptyStatements().visit(self.cfg)
    print_cfd(cfg=self.cfg, optimization_name="RemoveEmptyStatements")

    self.cfg = MergeActions().visit(self.cfg)
    print_cfd(cfg=self.cfg, optimization_name="MergeActions")


    
class KernelFamily(object):
  GROUP_INDEX = r'\((0|[1-9]\d*)\)'
  VALID_NAME = r'^{}({})$'.format(Kernel.BASE_NAME, GROUP_INDEX)

  def __init__(self):
    self._kernels = dict()
    self.name = None
    self._stride = None
  
  def items(self):
    return self._kernels.items()
  
  def __len__(self):
    return max(self._kernels.keys()) + 1
  
  @classmethod  
  def baseName(self, name):
    return re.match(Kernel.BASE_NAME, name).group(0)
  
  @classmethod
  def isValidName(cls, name):
    return re.match(cls.VALID_NAME, name) is not None
  
  @classmethod
  def group(cls, name):
    m = re.search(cls.GROUP_INDEX, name)
    return int(m.group(1))
  
  def setStride(self, stride):
    self._stride = stride
  
  def stride(self):
    if self._stride is not None:
      return self._stride
    return (1,)
    
  @classmethod
  def linear(cls, stride, group):
    assert len(stride) == len(group)
    index = 0
    for i,p in enumerate(group):
      index += p*stride[i]
    return index

  def add(self, name, ast, prefetch=None):
    baseName = self.baseName(name)
    if not self.name:
      self.name = baseName
    assert baseName == self.name
    
    group = self.group(name)
    internalName = '_{}_{}'.format(baseName, group)
    self._kernels[group] = Kernel(internalName, ast, prefetch)

  def kernels(self):
    return self._kernels.values()

  def prepareUntilUnitTest(self):
    for kernel in self._kernels.values():
      kernel.prepareUntilUnitTest()
  
  def prepareUntilCodeGen(self, costEstimator):
    for kernel in self._kernels.values():
      kernel.prepareUntilCodeGen(costEstimator)

def simpleParameterSpace(*args):
  return list(itertools.product(*[list(range(i)) for i in args]))

def parameterSpaceFromRanges(*args):
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
    self._kernels = list()
    self._kernelFamilies = dict()
    self._arch = arch

  def arch(self):
    return self._arch
  
  def add(self, name: str, ast: Node, prefetch=None):
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
    if name not in self._kernelFamilies:
      self._kernelFamilies[name] = KernelFamily()
    family = self._kernelFamilies[name]
    pmax = max(parameterSpace)
    stride = [1]
    for i in range(len(pmax)-1):
      stride.append(stride[i] * (pmax[i]+1))
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
    CUDA_SRC = 'cu'

    def __init__(self, outputDir, name):
      # names of header, source, cuda_source files
      self.hName = '{}.{}'.format(name, self.HEADER)
      self.cppName = '{}.{}'.format(name, self.CPP)
      self.cudaSrcName = '{}.{}'.format(name, self.CUDA_SRC)

      # paths to the source files
      self.h = os.path.join(outputDir, self.hName)
      self.cpp = os.path.join(outputDir, self.cppName)
      self.cu = os.path.join(outputDir, self.cudaSrcName)


  def __init__(self, arch):
    self._kernels = list()
    self._kernelFamilies = dict()
    self._arch = arch

  def arch(self):
    return self._arch

  def add(self, name: str, ast: Node, prefetch=None):
    if KernelFamily.isValidName(name):
      baseName = KernelFamily.baseName(name)
      if baseName not in self._kernelFamilies:
        self._kernelFamilies[baseName] = KernelFamily()
      self._kernelFamilies[baseName].add(name, ast, prefetch)
    else:
      if not Kernel.isValidName(name):
        raise ValueError(
          'Kernel name invalid (must match regexp {}): {}'.format(Kernel.VALID_NAME, name))
      kernel = Kernel(name, ast, prefetch)
      self._kernels.append(kernel)

  def kernels(self):
    return [kernel for kernel in self._kernels] + [kernel for family in
                                                   self._kernelFamilies.values() for kernel in
                                                   family.kernels()]

  def addFamily(self, name: str, parameterSpace, astGenerator, prefetchGenerator=None):
    if name not in self._kernelFamilies:
      self._kernelFamilies[name] = KernelFamily()
    family = self._kernelFamilies[name]
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

  def generate(self,
               outputDir: str,
               namespace='yateto',
               gemm_cfg: GeneratorCollection = None,
               costEstimator=BoundingBoxCostEstimator):

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
    with Cpp(fUT.cu) as unit_test_source:
      # specify all necessary hedear files which the unit tests depend on
      unit_test_source.includeSys('cxxtest/TestSuite.h')
      unit_test_source.include(fUT.hName)
      unit_test_source.include(fInit.hName)
      unit_test_source.include(fKernels.hName)
      unit_test_source.include(fTensors.hName)
      unit_test_source.include("cuda_utils.cuh")
      unit_test_source.include('Util.h')
      unit_test_source.include('TensorView.h')


      # generate the source code for unit tests
      for kernel in self._kernels:
        CudaUnitTestGenerator(self._arch).generate(unit_test_source,
                                                   kernel.name,
                                                   kernel.name,
                                                   kernel.cfg,
                                                   gemm_cfg,
                                                   function_namespace=full_test_class_name)

      for family in self._kernelFamilies.values():
        for group, kernel in family.items():
          CudaUnitTestGenerator(self._arch).generate(unit_test_source,
                                                     kernel.name,
                                                     family.name,
                                                     kernel.cfg,
                                                     gemm_cfg,
                                                     group)


    print('Optimizing ASTs...')
    for kernel in self._kernels:
      print(kernel.name)
      kernel.prepareUntilCodeGen(costEstimator)
    for family in self._kernelFamilies.values():
      print(family.name)
      family.prepareUntilCodeGen(costEstimator)


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

    with Cpp(fKernels.cu) as cpp:
      for gemm_tool in gemm_cfg.selected:

        for inc in gemm_tool.includes:
          cpp.include(inc)

        if isinstance(gemm_tool, BLASlike):
          cpp(gemm_tool.c_code_init)

      # add the kerlnel body generated above to a text source file
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
