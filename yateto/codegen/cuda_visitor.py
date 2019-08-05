import collections
import operator
from functools import reduce
from io import StringIO
from ..memory import DenseMemoryLayout
from ..controlflow.visitor import ScalarsSet, SortedGlobalsList, SortedPrefetchList
from ..controlflow.transformer import DetermineLocalInitialization
from ..controlflow.graph import Variable
from .code import Cpp


from .factory import *
from .visitor import KernelGenerator
from .cuda_factory import *


SUPPORT_LIBRARY_NAMESPACE = 'yateto'
CONSTEXPR = 'constexpr'
STATIC = 'static'
INLINE = 'inline'
MODIFIERS = '{} {}'.format(CONSTEXPR, STATIC)
STATIC_INLINE = '{} {}'.format(STATIC, INLINE)

def groupSizeToStride(groupSize):
  if len(groupSize) == 0:
    return tuple()
  stride = [1]
  for i in range(len(groupSize)-1):
    stride.append(stride[i] * groupSize[i])
  return tuple(stride)

def address(group, stride):
  return sum(map(operator.mul, group, stride))

def ndargs(d):
  return ['i' + str(i) for i in range(d)]

def typedNdArgs(d, uintTypename):
  typedArgs = ['{} {}'.format(uintTypename, arg) for arg in ndargs(d)]
  return ', '.join(typedArgs)

def indexFun(stride):
  if len(stride) == 0:
    return '0'
  args = ndargs(len(stride))
  return ' + '.join(['{}*{}'.format(stride, arg) for stride,arg in zip(stride,args)])

class CudaKernelGenerator(object):
  PREFETCHSTRUCT_NAME = 'Prefetch'
  PREFETCHVAR_NAME = '_prefetch'
  BUFFER_NAME = '_buffer'
  CUDA_BUFFER_NAME = 'd_buffer'


  def __init__(self, arch):
    """
    Args:
      arch (Architecture): TODO
    """
    self._arch = arch


  @classmethod
  def _bufferName(cls, buf):
    return cls.BUFFER_NAME + str(buf)


  @classmethod
  def _get_cuda_buffer_name(cls, buffer):
    return cls.CUDA_BUFFER_NAME + str(buffer)


  def generate(self, cpp, cfg, factory,  routineCache, gemm_cfg):
    """TODO

    Args:
      cpp (TODO): a file which a generated code is going to be written to
      cfg (List[ProgramPoint]): an execution block (a control flow graph)
      factory (Union[Type[KernelFactory], Type[CudaKernelFactory]]): TODO
      routineCache (RoutineCache): TODO
      gemm_cfg (Generator Collection): list of gemm generators

    Returns:
      int: an expected number of floating point operaions
    """
    hwFlops = 0
    cfg = DetermineLocalInitialization().visit(cfg)
    temp_pointers = list()

    # collect all pointer to temp variables
    for program_point in cfg:
      temp_pointers.extend(program_point.bufferMap.keys())

    # declare pointers to temp variables
    if temp_pointers:
      cpp('{}{};'.format(self._arch.typename,
                         ','.join(map(lambda x: ' *' + str(x), temp_pointers))))


    cpp('// TODO: allocate all buffers at the main entry point of the program')
    for program_point in cfg:


      # collect all buffers (scratch memory) and allocate memory for them
      # TODO: perform memory allocation of buffers at the program entry point
      for buffer, size in program_point.initBuffer.items():
        #buffer_name = self._bufferName(buffer)
        buffer_name = self._get_cuda_buffer_name(buffer)
        factory.cuda_temporary(buffer_name, size)


      for local, buffer in program_point.bufferMap.items():
        #cpp('{} = {};'.format(local, self._bufferName(buffer)))
        cpp('{} = {};'.format(local, self._get_cuda_buffer_name(buffer)))

      action = program_point.action
      if action:
        scalar = 1.0 if action.scalar is None else action.scalar
        if action.isRHSExpression():
          prefetchName = '{}.{}'.format(self.PREFETCHVAR_NAME,
                                        action.term.node.prefetch.name()) if action.term.node.prefetch is not None else None

          hwFlops += factory.create(action.term.node,
                                    action.result,
                                    action.term.variableList(),
                                    action.add,
                                    scalar,
                                    prefetchName,
                                    routineCache,
                                    gemm_cfg)
        else:
          hwFlops += factory.simple(action.result, action.term, action.add, scalar, routineCache)

    for program_point in cfg:
      for buffer, size in program_point.initBuffer.items():
        buffer_name = self._get_cuda_buffer_name(buffer)
        factory.cuda_delete_temporary(buffer_name)

    return hwFlops


class CudaOptimisedKernelGenerator(CudaKernelGenerator):
  NAMESPACE = 'kernel'
  EXECUTE_NAME = 'execute'
  FIND_EXECUTE_NAME = 'findExecute'
  EXECUTE_ARRAY_NAME = 'ExecutePtrs'
  NONZEROFLOPS_NAME = 'NonZeroFlops'
  HARDWAREFLOPS_NAME = 'HardwareFlops'
  MEMBER_FUNCTION_PTR_NAME = 'member_function_ptr'
  
  def __init__(self, arch, routineCache):
    """
    Args:
      arch (Architecture): TODO
      routineCache (RoutineCache): TODO
    """
    super().__init__(arch)
    self._routineCache = routineCache
  
  class KernelOutline(object):
    def __init__(self, nonZeroFlops, hwFlops, tensors, writable, prefetch, scalars, function):
      """
      Args:
        nonZeroFlops: TODO
        hwFlops: TODO
        tensors: TODO
        writable: TODO
        prefetch: TODO
        scalars: TODO
        function: TODO
      """
      self.nonZeroFlops = nonZeroFlops
      self.hwFlops = hwFlops
      self.tensors = tensors
      self.writable = writable
      self.prefetch = prefetch
      self.scalars = scalars
      self.function = function

    @classmethod
    def _addTensor(cls, tensor, tensors):
      """

      A table uses base tensor names as keys. Each key corresponds to group indices
      of all tensors involved in a kernel family. For example, given to tensor names 'A(0,1)'
      and 'A(1,1)' from a kernel family 'A'. A resultant table entry will the following have a form:
      table['A'] -> {(0,1), (1,1)}. In case of a general tensor i.e. which is not included into
      any family, for example 'C', the corresponding entry is going to look like:
      table['C'] = ()

      NOTE: the method is static

      Args:
        tensor (Type[Tensor]):
        tensors (OrderedDict[str, Set[Union[Typle[int, ..,], int]]]):

      Raises:
        ValueError:
      """
      base_name = tensor.baseName()
      group_index = tensor.group()


      if base_name in tensors:

        # take the first index of a kernel family as an example to check dimensions
        first = next(iter(tensors[base_name]))

        if len(first) != len(group_index):
          raise ValueError('Group size mismatch ({} vs {}) for {}.'.format(first,
                                                                           group_index,
                                                                           base_name))
        tensors[base_name] = tensors[base_name] | {group_index}
      else:
        tensors[base_name] = {group_index}


  def generateKernelOutline(self, nonZeroFlops, cfg, gemm_cfg):
    """TODO

    Args:
      nonZeroFlops (int): TODO: ask Carsten
      cfg (List[ProgramPoint]): control flow graph
      gemm_cfg (Generator Collection): list of gemm generators

    Returns:
      KernelOutline: TODO; NOTE: class KernelOutline is nested into [Cuda]OptimisedKernelGenerator
    """

    # iterate through a give control flow graph
    # and collect info about scalars and variables
    scalars = ScalarsSet().visit(cfg)  # Set[Union[Scalar]]
    variables = SortedGlobalsList().visit(cfg)  # List[Variable]

    tensors = collections.OrderedDict()  # OrderedDict[str, Set[Union[Typle[int, ..,], int]]]
    writable = dict()  # Dict[str, bool]

    # iterate through all global variable of an execution block (aka a control flow graph)
    for var in variables:
      # add tensors to a list
      # NOTE: _addTensor is a static method of class
      #        KernelOutline nested into [Cuda]OptimisedKernelGenerator
      self.KernelOutline._addTensor(var.tensor, tensors)


      # add a tensor to a table of 'writable' variables and assign
      # the bool value according to the content of the current variable
      base_name = var.tensor.baseName()
      if base_name in writable:
        if var.writable:
          writable[base_name] = True
      else:
        writable[base_name] = var.writable

    # generate a table of tensors that can be prefetched during a code execution
    # NOTE: a resultant list is always empty
    prefetchTensors = SortedPrefetchList().visit(cfg)
    prefetch = collections.OrderedDict()
    for tensor in prefetchTensors:
      self.KernelOutline._addTensor(tensor, prefetch)


    # generate C-code for tensor contraction and write to an StringIO instance
    functionIO = StringIO()
    function = ''  # str: an empty string which will be filled with TODO
    with Cpp(functionIO) as fcpp:

      # select a factory
      factory = CudaOptimisedKernelFactory(fcpp, self._arch)

      # generate C-code from a control from graph using a factory and a given GEMM method
      # NOTE:  fcpp is, in fact, functionIO
      hwFlops = super().generate(cpp=fcpp,
                                 cfg=cfg,
                                 factory=factory,
                                 routineCache=self._routineCache,
                                 gemm_cfg=gemm_cfg)

      # free temporary variables in case if there were allocate on the heap
      factory.freeTmp()

      # get a generated code as a text from the StringIO instance
      function = functionIO.getvalue()


    return self.KernelOutline(nonZeroFlops, hwFlops, tensors, writable, prefetch, scalars, function)


  @classmethod
  def _addFromKO(cls, koEntries, entries):
    for key, value in koEntries.items():
      if key not in entries:
        entries[key] = value
      else:
        entries[key] = entries[key] | value
    

  def generate(self, cpp, header, name, kernelOutlines, familyStride=None):
    """
    Generates both source and header files for a given kernel
    Args:
      cpp: a handle for the source file
      header: a handle for the source file
      name (str): name of a kernel
      kernelOutlines (List[kernelOutline]): TODO: ???
      familyStride: TODO: ???

    Returns:

    """
    tensors = collections.OrderedDict()
    prefetch = collections.OrderedDict()
    writable = dict()
    scalars = set()
    for outline in kernelOutlines:
      if outline:
        scalars = scalars | outline.scalars
        self._addFromKO(outline.tensors, tensors)
        self._addFromKO(outline.writable, writable)
        self._addFromKO(outline.prefetch, prefetch)

    scalars = sorted(list(scalars), key=str)

    if familyStride is not None:
      executeName = lambda index: self.EXECUTE_NAME + str(index)
      formatArray = lambda lst: '{{{}}}'.format(', '.join([str(l) for l in lst]))
      brackets = '[]'
    else:
      # TODO: ask what the purpose of these helper functions is
      executeName = lambda index: self.EXECUTE_NAME
      formatArray = lambda lst: lst[0]
      brackets = ''

    with header.Namespace(self.NAMESPACE):
      with header.Struct(name):

        # add declarations of profiling counters to a header file
        header('{} {} const {}{} = {};'.format(MODIFIERS,
                                               self._arch.ulongTypename,
                                               self.NONZEROFLOPS_NAME,
                                               brackets,
                                               formatArray([kernelOutline.nonZeroFlops
                                                            if kernelOutline else 0
                                                            for kernelOutline in kernelOutlines])))

        header('{} {} const {}{} = {};'.format(MODIFIERS,
                                               self._arch.ulongTypename,
                                               self.HARDWAREFLOPS_NAME,
                                               brackets,
                                               formatArray([kernelOutline.hwFlops
                                                            if kernelOutline else 0
                                                            for kernelOutline in kernelOutlines])))
        header.emptyline()


        for scalar in scalars:
          header('{0} {1} = std::numeric_limits<{0}>::signaling_NaN();'.format(self._arch.typename, scalar))


        # declare a helper function to generate a kernel header file
        def kernelArgs(baseName, groups, writable):
          typ = self._arch.typename
          if not writable:
            typ += ' const'
          if len(next(iter(groups))) > 0:
            header('{0}::{1}::{2}<{3}*> {1};'.format(CudaInitializerGenerator.TENSOR_NAMESPACE,
                                                     baseName,
                                                     CudaInitializerGenerator.CONTAINER_CLASS_NAME,
                                                     typ))
          else:
            header('{}* {}{{}};'.format(typ, baseName))

        # generate the body of a kernel header file: expose global kernel pointers
        for baseName, groups in tensors.items():
          kernelArgs(baseName, groups, writable[baseName])
        header.emptyline()


        if len(prefetch) > 0:
          with header.Struct(self.PREFETCHSTRUCT_NAME):
            for baseName, groups in prefetch.items():
              kernelArgs(baseName, groups, False)
          header('{} {};'.format(self.PREFETCHSTRUCT_NAME, self.PREFETCHVAR_NAME))
          header.emptyline()


        # add function declarations to the kernel header
        for index, kernelOutline in enumerate(kernelOutlines):
          if kernelOutline:
            header.functionDeclaration(executeName(index))


        # TODO: ask what familyStride is
        if familyStride is not None:
          header('typedef void ({}::* const {})(void);'.format(name, self.MEMBER_FUNCTION_PTR_NAME))
          header('{} {} {}[] = {};'.format(MODIFIERS,
                                           self.MEMBER_FUNCTION_PTR_NAME,
                                           self.EXECUTE_ARRAY_NAME,
                                           formatArray(['&{}::{}'.format(name, executeName(index))
                                                        if kernelOutline else 'nullptr'
                                                        for index, kernelOutline in enumerate(kernelOutlines)])))


          args = typedNdArgs(len(familyStride), self._arch.uintTypename)
          indexF = indexFun(familyStride)

          with header.Function(self.FIND_EXECUTE_NAME, args, '{} {}'.format(MODIFIERS,
                                                                            self.MEMBER_FUNCTION_PTR_NAME)):
            header('return {}[{}];'.format(self.EXECUTE_ARRAY_NAME, indexF))

          with header.Function(self.EXECUTE_NAME, args, '{} void'.format(INLINE)):
            header('(this->*{}({}))();'.format(self.FIND_EXECUTE_NAME, ', '.join(ndargs(len(familyStride)))))

          flopFuns = [self.NONZEROFLOPS_NAME, self.HARDWAREFLOPS_NAME]

          for flopFun in flopFuns:
            funName = flopFun[:1].lower() + flopFun[1:]
            with header.Function(funName, args, '{} {}'.format(MODIFIERS, self._arch.ulongTypename)):
              header('return {}[{}];'.format(flopFun, indexF))


    # declare flop counter variables in the kernel source file
    flopCounters = [self.NONZEROFLOPS_NAME, self.HARDWAREFLOPS_NAME]
    for counter in flopCounters:
      cpp('{} {} const {}::{}::{}{};'.format(CONSTEXPR,
                                             self._arch.ulongTypename,
                                             self.NAMESPACE,
                                             name,
                                             counter,
                                             brackets))

    if familyStride is not None:
      cpp('{0} {1}::{2}::{3} {1}::{2}::{4}[];'.format(CONSTEXPR,
                                                      self.NAMESPACE,
                                                      name,
                                                      self.MEMBER_FUNCTION_PTR_NAME,
                                                      self.EXECUTE_ARRAY_NAME))


    for index, kernelOutline in enumerate(kernelOutlines):
      if kernelOutline is None:
        continue

      # generate a kernel body
      with cpp.Function('{}::{}::{}'.format(self.NAMESPACE, name, executeName(index))):
        scalars = sorted(list(kernelOutline.scalars), key=str)

        # check whether scalars used in a tensor equation
        # were assigned to some numerical values
        for scalar in scalars:
          cpp('assert(!std::isnan({}));'.format(scalar))


        # iterate through each tensor defined in an equation
        for baseName, groups in kernelOutline.tensors.items():

          if len(next(iter(groups))) > 0:
            # TODO: ask what gis stands for
            for gis in groups:

              # check whether tensor pointers are not equal to zero
              # i.e. points to some chunck of memory
              # TODO: ask what gi stands for
              cpp('assert({}({}) != nullptr);'.format(baseName, ','.join(str(gi) for gi in gis)))

          else:
            cpp('assert({} != nullptr);'.format(baseName))

        # generate kernel logic
        cpp(kernelOutline.function)


class CudaUnitTestGenerator(CudaKernelGenerator):
  KERNEL_VAR = 'krnl'
  CXXTEST_PREFIX = 'test'

  def __init__(self, arch):
    super().__init__(arch)

  @classmethod
  def _tensorName(cls, var):
    if var.isLocal():
      return str(var)
    baseName = var.tensor.baseName()
    group = var.tensor.group()
    terms = [baseName] + [str(g) for g in group]
    return '_'.join(terms)

  @classmethod
  def _get_gpu_tensor_name(cls, var):
    prefix = "d_"

    if var.isLocal():
      return prefix + str(var)

    baseName = prefix + var.tensor.baseName()

    group = var.tensor.group()
    terms = [baseName] + [str(index) for index in group]
    return '_'.join(terms)


  @classmethod
  def _name(cls, var):
    if var.isLocal():
      return str(var)
    return '_ut_' + cls._tensorName(var)

  def _viewName(self, var):
    return '_view_' + self._name(var)
  
  def _groupStr(self, var):
    group = var.tensor.group()
    return ','.join([str(g) for g in group])

  def _groupTemplate(self, var):
    gstr = self._groupStr(var)
    return '<{}>'.format(gstr) if gstr else ''

  def _groupIndex(self, var):
    gstr = self._groupStr(var)
    return '({})'.format(gstr) if gstr else ''
  
  def generate(self, cpp, testName, kernelClass, cfg, gemm_cfg, index=None, function_namespace=""):
    scalars = ScalarsSet().visit(cfg)
    scalars = sorted(scalars, key=str)
    variables = SortedGlobalsList().visit(cfg)


    if function_namespace:
      function_name = function_namespace + "::" + self.CXXTEST_PREFIX + testName
    else:
      function_name = self.CXXTEST_PREFIX + testName

    print('\n'*3)
    with cpp.Function(function_name):
      cuda_factory = CudaUnitTestFactory(cpp, self._arch, self._name)
      cpu_factory = UnitTestFactory(cpp, self._arch, self._name)

      for i, scalar in enumerate(scalars):
        cpp('{} {} = {};'.format(self._arch.typename, scalar, float(i+2)))
        
      for var in variables:
        # allocate and init tensors on CPU
        cuda_factory.tensor(var.tensor, self._tensorName(var))
        cuda_factory.temporary(self._name(var),
                               var.memoryLayout().requiredReals(),
                               iniZero=True)

        shape = var.memoryLayout().shape()
        cpp('{supportNS}::DenseTensorView<{dim},{arch.typename},{arch.uintTypename}> {viewName}({utName}, {{{shape}}}, {{{start}}}, {{{shape}}});'.format(
            supportNS = SUPPORT_LIBRARY_NAMESPACE,
            dim=len(shape),
            arch = self._arch,
            utName=self._name(var),
            viewName=self._viewName(var),
            shape=', '.join([str(s) for s in shape]),
            start=', '.join(['0']*len(shape))
          )
        )
        cpp( '{initNS}::{baseName}::{viewStruct}{groupTemplate}::{createFun}({name}).copyToView({viewName});'.format(
            initNS = CudaInitializerGenerator.INIT_NAMESPACE,
            supportNS = SUPPORT_LIBRARY_NAMESPACE,
            groupTemplate=self._groupTemplate(var),
            baseName=var.tensor.baseName(),
            name=self._tensorName(var),
            viewName=self._viewName(var),
            viewStruct=CudaInitializerGenerator.VIEW_STRUCT_NAME,
            createFun=CudaInitializerGenerator.VIEW_FUN_NAME
          )
        )
        cpp.emptyline()

      cpp('{}::{} {};'.format(CudaOptimisedKernelGenerator.NAMESPACE,
                              kernelClass,
                              self.KERNEL_VAR))

      for scalar in scalars:
        cpp( '{0}.{1} = {1};'.format(self.KERNEL_VAR, scalar))


      for var in variables:
        cpp( '{0}.{1}{2} = {3};'.format(self.KERNEL_VAR,
                                        var.tensor.baseName(),
                                        self._groupIndex(var),
                                        self._get_gpu_tensor_name(var)))

      # execute a kernel
      cpp( '{}.{}();'.format(self.KERNEL_VAR,
                             CudaOptimisedKernelGenerator.EXECUTE_NAME + (str(index) if index is not None else '')))
      cpp.emptyline()


      # copy result of execution from GPU to CPU
      # iterate through all variables in the control flow graph
      for var in variables:
        # consider only tensors which a writable
        # i.e. modified during the kernel execution
        if var.tensor:
          if var.writable:
            memory_layout = var.tensor.memoryLayout()
            size = memory_layout.requiredReals()
            size_in_bytes = "sizeof({0}) * {1}".format(self._arch.typename,
                                                       size)

            function_params = "{0}, {1}, {2}, {3}".format(self._tensorName(var),
                                                          self._get_gpu_tensor_name(var),
                                                          size_in_bytes,
                                                          "cudaMemcpyDeviceToHost")

            cpp('cudaMemcpy({}); CUDA_CHECK;'.format(function_params))
          cpp('cudaFree({}); CUDA_CHECK;'.format(self._get_gpu_tensor_name(var)))


      cpp.emptyline()


      # use CPU kernek generator to generate a unit test
      KernelGenerator(self._arch).generate(cpp, cfg, cpu_factory, None, gemm_cfg)


      for var in variables:
        if var.writable:
          cpu_factory.compare(var, Variable(self._tensorName(var), False, var.tensor.memoryLayout()))

      cpu_factory.freeTmp()


class CudaInitializerGenerator(object):
  SHAPE_NAME = 'Shape'
  SIZE_NAME = 'Size'
  SIZE_FUN_NAME = 'size'
  INDEX_FUN_NAME = 'index'
  VALUES_BASENAME = 'Values'
  CONTAINER_CLASS_NAME = 'Container'
  CONTAINER_DATA_NAME = 'data'
  TENSOR_NAMESPACE = 'tensor'
  INIT_NAMESPACE = 'init'
  VIEW_STRUCT_NAME = 'view'
  VIEW_FUN_NAME = 'create'
  VIEW_TYPE_NAME = 'type'
  
  class TensorView(object):
    ARGUMENT_NAME = 'values'

    def typename(self, dim, arch):
      return '::{}::{}<{},{},{}>'.format(SUPPORT_LIBRARY_NAMESPACE, type(self).__name__, dim, arch.typename, arch.uintTypename)
    
    @classmethod
    def arguments(cls, arch):
      return '{}* {}'.format(arch.typename, cls.ARGUMENT_NAME)
    
    def generate(cpp, group, memLayout):
      raise NotImplementedError
    
    def listToInitializerList(self, lst):
      return '{{{}}}'.format(', '.join([str(l) for l in lst]))
    
    def formatArray(self, numberType, name, values, declarationOnly):
      lhs = '{} {}[]'.format(numberType, name)
      if declarationOnly:
        return '{} {};'.format(CONSTEXPR, lhs)
      return '{} {} = {};'.format(MODIFIERS, lhs, self.listToInitializerList(values))
  
  class DenseTensorView(TensorView):
    START_NAME = 'Start'
    STOP_NAME = 'Stop'

    def generate(self, cpp, memLayout, arch, index):
      cpp( 'return {}({}, {}, {}, {});'.format(
          self.typename(len(memLayout.shape()), arch),
          self.ARGUMENT_NAME,
          self.listToInitializerList(memLayout.shape()),
          self.listToInitializerList([r.start for r in memLayout.bbox()]),
          self.listToInitializerList([r.stop for r in memLayout.bbox()])
        )
      )
    def arrays(self, cpp, memLayout, arch, namespace, index, numberType, declarationOnly):
      cpp(self.formatArray(numberType, namespace + self.START_NAME + index, [r.start for r in memLayout.bbox()], declarationOnly))
      cpp(self.formatArray(numberType, namespace + self.STOP_NAME + index, [r.stop for r in memLayout.bbox()], declarationOnly))


  class CSCMatrixView(TensorView):
    ROWIND_NAME = 'RowInd'
    COLPTR_NAME = 'ColPtr'
    
    def typename(self, dim, arch):
      return '::{}::{}<{},{}>'.format(SUPPORT_LIBRARY_NAMESPACE, type(self).__name__, arch.typename, arch.uintTypename)

    def generate(self, cpp, memLayout, arch, index):
      cpp( 'return {}({}, {}, {}, {});'.format(
          self.typename(len(memLayout.shape()), arch),
          self.ARGUMENT_NAME,
          self.listToInitializerList(memLayout.shape()),
          self.ROWIND_NAME + (index if index is not None else ''),
          self.COLPTR_NAME + (index if index is not None else '')
        )
      )
    def arrays(self, cpp, memLayout, arch, namespace, index, numberType, declarationOnly):
      cpp(self.formatArray(numberType, namespace + self.ROWIND_NAME + index, memLayout.rowIndex(), declarationOnly))
      cpp(self.formatArray(numberType, namespace + self.COLPTR_NAME + index, memLayout.colPointer(), declarationOnly))

  def __init__(self, arch, tensors):
    self._arch = arch
    self._numberType = '{} const'.format(self._arch.uintTypename)
    self._realType = '{} const'.format(self._arch.typename)
    self._realPtrType = self._realType + '*'
    self._collect = collections.OrderedDict()

    for tensor in tensors:
      baseName = tensor.baseName()
      group = tensor.group()
      if baseName not in self._collect:
        self._collect[baseName] = {group: tensor}
      elif group not in self._collect[baseName]:
        groupRef = next(iter(self._collect[baseName].keys()))
        if len(group) != len(groupRef):
          raise ValueError('Mixed group dimensions are not allowed. ({} and {} for {}.)'.format(group, groupRef, baseName))
        self._collect[baseName][group] = tensor
      else:
        assert self._collect[baseName][group] == tensor
    maxIndex = {baseName: tuple(map(max, *groups.keys())) if len(groups) > 1 else next(iter(groups.keys())) for baseName, groups in self._collect.items()}
    self._groupSize = {baseName: tuple(map(lambda x: x+1, mi)) for baseName, mi in maxIndex.items()}


  def _tensorViewGenerator(self, memoryLayout):
    memLayoutMap = {
      'DenseMemoryLayout': self.DenseTensorView,
      'CSCMemoryLayout': self.CSCMatrixView
    }
    return memLayoutMap[type(memoryLayout).__name__]()
  
  def generateTensorsH(self, header):
    with header.Namespace(self.TENSOR_NAMESPACE):

      header("// DEBUG: CUDA part")
      header("extern {} num_elements_in_cluster;".format(self._arch.uintTypename))
      header.emptyline()

      for baseName, tensors in self._collect.items():

        with header.Struct(baseName):
          groupSize = self._groupSize[baseName]
          self._tensor(header, '', tensors, groupSize, False)
          args = ndargs(len(groupSize))
          typedArgs = typedNdArgs(len(groupSize), self._arch.uintTypename)
          returnType = '{} {}'.format(MODIFIERS, self._arch.uintTypename)

          if len(groupSize) > 0:
            with header.Function(self.INDEX_FUN_NAME, typedArgs, returnType):
              header('return {};'.format(indexFun(groupSizeToStride(groupSize))))

          with header.Function(self.SIZE_FUN_NAME, typedArgs, returnType):
            if len(groupSize) == 0:
              header('return {};'.format(self.SIZE_NAME))
            else:
              header('return {}[{}({})];'.format(self.SIZE_NAME, self.INDEX_FUN_NAME, ', '.join(args)))

          if len(groupSize) > 0:
            header('template<typename T>')
            with header.Struct(self.CONTAINER_CLASS_NAME):
              header('T {}[{}];'.format(self.CONTAINER_DATA_NAME, reduce(operator.mul, groupSize)))
              header('{}() : {}{{}} {{}}'.format(self.CONTAINER_CLASS_NAME, self.CONTAINER_DATA_NAME))
              with header.Function('operator()', typedArgs, '{} T&'.format(INLINE)):
                header('return {}[{}({})];'.format(self.CONTAINER_DATA_NAME, self.INDEX_FUN_NAME, ', '.join(args)))
              with header.Function('operator()', typedArgs, '{} T const&'.format(INLINE), const=True):
                header('return {}[{}({})];'.format(self.CONTAINER_DATA_NAME, self.INDEX_FUN_NAME, ', '.join(args)))


  def generateTensorsCpp(self, cpp):
    cpp("// DEBUG: CUDA part (where 1 is a default value)")
    cpp("{} {}::num_elements_in_cluster = 1;".format(self._arch.uintTypename,
                                                       self.TENSOR_NAMESPACE))
    cpp.emptyline()


    for baseName, tensors in self._collect.items():
      self._tensor(cpp=cpp,
                   name='::'.join([self.TENSOR_NAMESPACE, baseName, '']),
                   tensors=tensors,
                   groupSize=self._groupSize[baseName],
                   declarationOnly=True)


  def generateInitH(self, header):
    with header.Namespace(self.INIT_NAMESPACE):
      for baseName, tensors in self._collect.items():
        self._init(cpp=header,
                   baseName=baseName,
                   name='',
                   tensors=tensors,
                   declarationOnly=False)


  def generateInitCpp(self, header):
    for baseName, tensors in self._collect.items():
      self._init(cpp=header,
                 baseName=baseName,
                 name='::'.join([self.INIT_NAMESPACE, baseName, '']),
                 tensors=tensors,
                 declarationOnly=True)


  def _tensor(self, cpp, name, tensors, groupSize, declarationOnly):
    # specify tensor shape
    shape = {group: tensor.shape() for group, tensor in tensors.items()}
    self._array(cpp=cpp,
                type=self._numberType,
                name=name + self.SHAPE_NAME,
                content=shape,
                groupSize=groupSize,
                declarationOnly=declarationOnly)


    # specify tensor size
    size = {group: [tensor.memoryLayout().requiredReals()] for group, tensor in tensors.items()}
    self._array(cpp=cpp,
                type=self._numberType,
                name=name + self.SIZE_NAME,
                content=size,
                groupSize=groupSize,
                declarationOnly=declarationOnly,
                alwaysArray=False)

    # specify jump to the next element
    # The jump is equal to zero if tensor is 'constant' i.e. the same for all elements
    # The jump is equal to a tensor size in case if a tensor is unique for each element
    default_jump = 0
    jump_to_next = {group: [default_jump] for group, tensor in tensors.items()}

    self._array(cpp=cpp,
                type=self._numberType,
                name=name + "jump_to_next",
                content=jump_to_next,
                groupSize=groupSize,
                declarationOnly=declarationOnly,
                alwaysArray=False,
                constexpr=True)


  def _init(self, cpp, baseName, name, tensors, declarationOnly):
    groupSize = self._groupSize[baseName]
    stride = groupSizeToStride(groupSize)
    index = lambda group: str(address(group, stride)) if len(group) > 0 else ''

    if declarationOnly:
      for group,tensor in tensors.items():
        ml = tensor.memoryLayout()
        tv = self._tensorViewGenerator(ml)
        tv.arrays(cpp, ml, self._arch, name, index(group), self._numberType, True)
      valueNames = dict()
      for group,tensor in tensors.items():
        values = tensor.values()
        memLayout = tensor.memoryLayout()
        if values is not None:
          memory = ['0.']*memLayout.requiredReals()
          for idx,x in values.items():
            memory[memLayout.address(idx)] = x
          valuesName = '{}{}{}'.format(name, self.VALUES_BASENAME, index(group))
          valueNames[group] = ['&{}[0]'.format(valuesName)]
          cpp('{} {}[] = {{{}}};'.format(self._realType, valuesName, ', '.join(memory)))
      if len(valueNames) > 1:
        self._array(cpp, self._realPtrType, name + self.VALUES_BASENAME, valueNames, groupSize, alwaysArray=False, constexpr=False, static=False)
    else:
      with cpp.Struct('{0} : {1}::{0}'.format(baseName, self.TENSOR_NAMESPACE)):
        for group,tensor in tensors.items():
          ml = tensor.memoryLayout()
          tv = self._tensorViewGenerator(ml)
          tv.arrays(cpp, ml, self._arch, name, index(group), self._numberType, False)

        nValueArrays = 0
        for group,tensor in tensors.items():
          values = tensor.values()
          if values is not None:
            name = '{}{}'.format(self.VALUES_BASENAME, index(group))
            aligned = ''
            if tensor.memoryLayout().alignedStride():
              aligned = ' __attribute__((aligned({})))'.format(self._arch.alignment)
            cpp('{} {} {}[]{};'.format(STATIC, self._realType, name, aligned))
            nValueArrays += 1
        if nValueArrays > 1:
          cpp('{} {} {}[];'.format(STATIC, self._realPtrType, self.VALUES_BASENAME))

        cpp.emptyline()
        viewArgs = self.TensorView.arguments(self._arch)
        if len(groupSize) == 0:
          ml = next(iter(tensors.values())).memoryLayout()
          tv = self._tensorViewGenerator(ml)
          with cpp.Struct(self.VIEW_STRUCT_NAME):
            cpp('typedef {} {};'.format(tv.typename(len(ml.shape()), self._arch), self.VIEW_TYPE_NAME))
            with cpp.Function(self.VIEW_FUN_NAME, arguments=viewArgs, returnType='{} {}'.format(STATIC_INLINE, self.VIEW_TYPE_NAME)):
              tv.generate(cpp, ml, self._arch, None)
        else:
          typedArgs = typedNdArgs(len(groupSize), self._arch.uintTypename)
          cpp('template<{}> struct {} {{}};'.format(typedArgs, self.VIEW_STRUCT_NAME))

      if len(groupSize) > 0:
        for group,tensor in tensors.items():
          ml = tensor.memoryLayout()
          tv = self._tensorViewGenerator(ml)
          typename = tv.typename(len(ml.shape()), self._arch)
          special = ','.join(str(g) for g in group)
          cpp('template<>')
          with cpp.Struct('{}::{}<{}>'.format(baseName, self.VIEW_STRUCT_NAME, special)):
            cpp('typedef {} {};'.format(typename, self.VIEW_TYPE_NAME))
            with cpp.Function(self.VIEW_FUN_NAME, arguments=viewArgs, returnType='{} {}'.format(STATIC_INLINE, self.VIEW_TYPE_NAME)):
              tv.generate(cpp, ml, self._arch, index(group))


  def _array(self, cpp, type, name, content, groupSize, declarationOnly=False, alwaysArray=True, constexpr=True, static=True):
    cexpr = CONSTEXPR + ' ' if constexpr else ''
    stat = STATIC + ' ' if static else ''
    maxLen = max(map(len, content.values())) if len(content.values()) > 0 else 0

    isGroup = len(groupSize) > 0
    groupIndices = '[]' if isGroup else ''

    isArray = alwaysArray or maxLen > 1
    arrayIndices = '[{}]'.format(maxLen) if isArray else ''
    
    if declarationOnly:
      cpp('{}{} {}{}{};'.format(cexpr, type, name, groupIndices, arrayIndices))
    else:
      formatArray = lambda L: ', '.join([str(x) for x in L])
      if isGroup:
        stride = groupSizeToStride(groupSize)
        size = reduce(operator.mul, groupSize, 1)
        init = ['0']*size
        for key, value in content.items():
          idx = address(key, stride)
          init[idx] = formatArray(value)
      else:
        init = [formatArray(next(iter(content.values())))]

      if isArray:
        init = ['{{{}}}'.format(i) for i in init]
      initStr = ', '.join(init)
      if isGroup:
        initStr = '{{{}}}'.format(initStr)
      
      cpp('{}{}{} {}{}{} = {};'.format(cexpr, stat, type, name, groupIndices, arrayIndices, initStr))
