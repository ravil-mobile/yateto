#!/usr/bin/env python3

import sys
sys.path.append('..')

import os
import errno
import argparse
import importlib.util
from yateto import *
from yateto.ast.visitor import PrettyPrinter, FindTensors, PrintEquivalentSparsityPatterns
from yateto.controlflow.visitor import PrettyPrinter as CfgPrinter
from yateto.codegen.code import Cpp


# read and parse the command line input
cmdLineParser = argparse.ArgumentParser()
cmdLineParser.add_argument('--arch',
                           type=str,
                           default='dhsw',
                           help='Architecture (e.g. dsnb for double precision on Sandy Bridge).')

cmdLineParser.add_argument('--variant',
                           type=str,
                           default='',
                           help='Example specific variant (e.g. onlyblas).')

cmdLineParser.add_argument('example_script',
                           type=str,
                           help='A yateto example script from the examples folder (without file extension).')
cmdLineArgs = cmdLineParser.parse_args()

# import the user's script containing tensor equations
exampleSpec = importlib.util.find_spec(cmdLineArgs.example_script)
try:
  example = exampleSpec.loader.load_module()
except:
  raise RuntimeError('Could not find example ' + cmdLineArgs.example_script)

targetFlopsPerSec = 40.0e9


# create the output directory
variantSuffix = '_' + cmdLineArgs.variant if cmdLineArgs.variant else ''
outDir = os.path.join(cmdLineArgs.example_script + "-gpu", cmdLineArgs.arch + variantSuffix)
try:
  if os.path.isdir(outDir):
    os.remove(outDir)

  os.makedirs(outDir)
except OSError as e:
  if e.errno == errno.EEXIST:
    pass


# create an architecture object based on the command line parameters
arch = useArchitectureIdentifiedBy(cmdLineArgs.arch)

# init generator according to the specified architecture
generator = CudaGenerator(arch)


# call the user's script where he/she defines a desired contraction
# with yateto DSL. The function call builds AST from an equation
example.add(generator)


# decide which GEMM generator to use i.e. either provided by the user of the default one
gemm_cfg = example.gemm_cfg(arch, variant='cuda') if hasattr(example, 'gemm_cfg') else None


# generate kernel i.e. translate from yateto to cpp
generator.generate(outDir, gemm_cfg=gemm_cfg)


# print out AST on the screen
print("{0} AST {0}".format("-"*30))
for kernel in generator.kernels():
  title = 'AST of {}'.format(kernel.name)
  print(title)
  print('='*len(title))
  PrettyPrinter().visit(kernel.ast)
  print(' ')


# print out CFG on the screen
print("{0} CFG {0}".format("-"*30))
for kernel in generator.kernels():
  CfgPrinter().visit(kernel.cfg)


# TODO: ask Carsten about this functionality
printEqspp = example.printEqspp() if hasattr(example, 'printEqspp') else False
if printEqspp:
  for kernel in generator.kernels():
    d = os.path.join(outDir, kernel.name)
    os.makedirs(d, exist_ok=True)
    PrintEquivalentSparsityPatterns(d).visit(kernel.ast)


# declare helper function which simplify code generation (see bellow)
formatArrayName = lambda tensor: '{0}__{1}'.format(tensor.baseName(), '_'.join([str(g) for g in tensor.group()]))
formatArrayCudaName = lambda tensor: 'd_{0}__{1}'.format(tensor.baseName(), '_'.join([str(g) for g in tensor.group()]))
formatGroup = lambda tensor: ','.join([str(g) for g in tensor.group()])


trashTheCache = example.cold() if hasattr(example, 'cold') else False
trashSize = 128 * 1024**2 # 128 MB to trash the cache

# generate an executable which is supposed to evict cache before profiling
with Cpp(os.path.join(outDir, 'trashTheCache.cpp')) as cpp:
  with cpp.Function('trashTheCache', arguments='double* trash, int size'):
        with cpp.For('int i = 0; i < size; ++i'):
          cpp('trash[i] += trash[i];')


# generate the main file which contains the entry point
with Cpp(os.path.join(outDir, 'performance.cu')) as cpp:

  # generate includes
  cpp.includeSys('cstdlib')
  cpp.includeSys('cstdio')
  cpp.includeSys('cmath')
  cpp.include('kernel.h')
  cpp.include('tensor.h')
  cpp.include('Stopwatch.h')
  cpp.include('Util.h')
  cpp.include('cuda_utils.cuh')

  cpp('using namespace yateto;')
  cpp.functionDeclaration('trashTheCache', arguments='double* trash, int size')

  with cpp.Function('main', arguments='int argc, char** argv', returnType='int'):

    # read command line input
    cpp('int _fixedReps = (argc >= 2) ? atoi(argv[1]) : -1;')
    cpp('int _reps, _error;')

    # allocate helper variables
    if trashTheCache:
      cpp('double* _trash = new double[{}];'.format(trashSize))

    cpp('Stopwatch _sw;')
    cpp('double _time, _nzflops, _flops;')
    cpp('printf("kernel,repetitions,time,numnzflop,numflop,nzgflops,gflops\\n");')

    # generate cpp body for all generated kernels
    for kernel in generator.kernels():
      with cpp.AnonymousScope():
        tensors = FindTensors().visit(kernel.ast).items()

        # allocate array for tensors on CPU
        cpp.emptyline()
        for key, tensor in tensors:
          cpu_arrayName = formatArrayName(tensor)
          cpp('real* {};'.format(cpu_arrayName))
          cpp('_error = posix_memalign(reinterpret_cast<void**>(&{0}), '
              'ALIGNMENT, tensor::{1}::size({2}) * sizeof(real));'.format(cpu_arrayName,
                                                                          tensor.baseName(),
                                                                          formatGroup(tensor)))


        # allocate arrays for tensors on GPU
        cpp.emptyline(num_lines=2)
        cpp('// allocated data on GPU')
        for key, tensor in tensors:
          gpu_arrayName = formatArrayCudaName(tensor)
          cpp('real* {};'.format(gpu_arrayName))

        for key, tensor in tensors:
          gpu_arrayName = formatArrayCudaName(tensor)
          cpp('cudaMalloc(&{0}, tensor::{1}::size({2}) * sizeof(real)); CUDA_CHECK;'.format(
                gpu_arrayName,
                tensor.baseName(),
                formatGroup(tensor)))


        # fill CPU tensor with random numbers
        cpp.emptyline(num_lines=2)
        for key,tensor in tensors:
          cpp('fillWithStuff({0}, tensor::{1}::size({2}));'.format(formatArrayName(tensor),
                                                                   tensor.baseName(),
                                                                   formatGroup(tensor)))
        # copy data from CPU tensors to GPU ones
        cpp.emptyline(num_lines=2)
        cpp('// move data from CPU to GPU')
        for key, tensor in tensors:
          gpu_arrayName = formatArrayCudaName(tensor)
          cpu_arrayName = formatArrayName(tensor)
          cpp('cudaMemcpy({0}, {1}, tensor::{2}::size({3}) * sizeof(real), '
              'cudaMemcpyHostToDevice); CUDA_CHECK;'.format(gpu_arrayName,
                                                            cpu_arrayName,
                                                            tensor.baseName(),
                                                            formatGroup(tensor)))
        cpp.emptyline()

        cpp("// TODO: allocate pointers for temp variables and scratch buffers on GPU")
        """
        for program_point in kernel.cfg:
          print("WE ARE HERE")

          for buffer, size in program_point.initBuffer.items():
            # buffer_name = self._bufferName(buffer)
            # buffer_name = self._get_cuda_buffer_name(buffer)
            # factory.cuda_temporary(buffer_name, size)
            print(buffer, size)
        """


        # check gpu operation
        cpp.emptyline()
        cpp("gpu_operation_check();")

        # adjust number of repetitions
        cpp.emptyline()
        if trashTheCache:
          cpp('_reps = 1;')
        else:
          cpp('_reps = _fixedReps;')

          with cpp.If('_reps < 0'):
            #cpp('_reps = ceil({0}/kernel::{1}::HardwareFlops);'.format(targetFlopsPerSec, kernel.name))
            cpp("// DEBUG: set up a limited number of repetitions for development")
            cpp('_reps = 1000;')

        cpp.emptyline(num_lines=2)
        kobj = '_kernel_{0}'.format(kernel.name)


        # assign allocated arrays to the pointers that a kernel is going to use
        cpp('kernel::{} {};'.format(kernel.name, kobj))
        for key, tensor in tensors:
          cpp('{0}.{1} = {2};'.format(kobj, key, formatArrayCudaName(tensor)))

        # start execution of the main loop and start measuring execution time
        if trashTheCache:
          cpp('trashTheCache(_trash, {});'.format(trashSize))
          cpp('_sw.start();')
          cpp('{}.execute();'.format(kobj))
        else:
          cpp('{}.execute();'.format(kobj))
          cpp('_sw.start();')
          with cpp.For('int i = 0; i < _reps; ++i'):
            cpp('{}.execute();'.format(kobj))
        cpp('_time = _sw.stop();')

        # compute profiling results
        cpp('_nzflops = static_cast<double>(kernel::{0}::NonZeroFlops) * _reps / _time / 1.0e9;'.format(kernel.name))
        cpp('_flops = static_cast<double>(kernel::{0}::HardwareFlops) * _reps / _time / 1.0e9;'.format(kernel.name))

        # print out results of profiling
        cpp('printf("{0},%u,%lf,%lu,%lu,%lf,%lf\\n", '
            '_reps, _time, kernel::{0}::NonZeroFlops, '
            'kernel::{0}::HardwareFlops, _nzflops, _flops);'.format(kernel.name))



        # copy data from CPU tensors to GPU ones
        cpp.emptyline(num_lines=2)


        # TODO: find API which can do the same job
        computation_results = set()
        for program_point in kernel.cfg:

          # check wether a programming point has an action
          if program_point.action is not None:

            if program_point.action.result.tensor and program_point.action.result.writable:
              gpu_arrayName = formatArrayCudaName(program_point.action.result.tensor)
              cpu_arrayName = formatArrayName(program_point.action.result.tensor)

              # prevent multiple occurances of the same tensor
              if cpu_arrayName not in computation_results:
                cpp('cudaMemcpy({0}, {1}, tensor::{2}::size({3}) * sizeof(real), '
                    'cudaMemcpyDeviceToHost); CUDA_CHECK;'.format(cpu_arrayName,
                                                                  gpu_arrayName,
                                                                  tensor.baseName(),
                                                                  formatGroup(tensor)))
                computation_results.add(cpu_arrayName)


        # free memory allocated for Tensors on GPU
        cpp.emptyline(num_lines=2)
        for key, tensor in tensors:
          cpp('cudaFree({}); CUDA_CHECK;'.format(formatArrayCudaName(tensor)))


        # free memory allocated for Tensors on CPU
        cpp.emptyline(num_lines=2)
        for key, tensor in tensors:
          cpp('free({});'.format(formatArrayName(tensor)))


    if trashTheCache:
      cpp('delete[] _trash;')
    cpp('return 0;')
