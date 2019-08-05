from ...ast.indices import Indices
from ..common import *
from .. import gemm

from yateto.type import Tensor

import re

class Generic(object):
  def __init__(self, arch, descr):
    self._arch = arch
    self._descr = descr


  def _pointer(self, cpp, targetName, baseName, term, loopIndices, const=True):
    indices = term.indices & loopIndices
    addressStr = term.memoryLayout.addressString(term.indices, indices) if len(indices) > 0 else ''
    if len(addressStr) > 0:
      addressStr = ' + ' + addressStr

    cpp('{} {}* {} = {}{};'.format(self._arch.typename,
                                   'const' if const else '',
                                   targetName,
                                   baseName,
                                   addressStr))



  def _generate_tensors_jumps(self, cpp, tensor_descriptions):

    terms = {"A": tensor_descriptions.leftTerm,
             "B": tensor_descriptions.rightTerm,
             "C": tensor_descriptions.result}

    jump_table = {}

    temp_variable_name = re.compile(r'_tmp*')
    base_jump_name = "jump_to_next_"
    for label, term in terms.items():

      size = term.eqspp.size
      if not temp_variable_name.match(term.name):
        size = "tensor::{}::jump_to_next".format(Tensor.getBaseName(term.name))

        if Tensor.getGroup(term.name):

          # in case if a tensor belongs to a tensor group
          size = size + "[{}]".format(*Tensor.getGroup(term.name))

        cpp("const {} {} = {};".format(self._arch.uintTypename,
                                       base_jump_name + Tensor.getBaseName(term.name),
                                       size))
        jump_table[label] = base_jump_name + Tensor.getBaseName(term.name)
      else:
        cpp("const {} {} = {};".format(self._arch.uintTypename,
                                       base_jump_name + term.name,
                                       size))

        jump_table[label] = base_jump_name + term.name


    cpp.emptyline()

    return jump_table


  def _alignedStart(self, term, loopIndices):
    if len(loopIndices) == 0:
      return True
    return term.memoryLayout.isAlignedAddressString(term.indices, term.indices & loopIndices)


  def _memLayout(self, term, I, J):
    assert len(I) > 0
    if len(J) == 0:
      ml = term.memoryLayout.vec(term.indices, I)
      return ml.withDummyDimension()
    elif len(term.indices) == 2:
      return term.memoryLayout
    return term.memoryLayout.unfold(term.indices, I, J)


  def _reduce(self, term, subset, memLayout):
    return reduceSpp(term.eqspp, term.indices, subset).reshape(memLayout.shape())


  def _defuse(self, fusedRange, term, I):
    if len(I) == 1:
      return  {next(iter(I)): fusedRange}
    return term.memoryLayout.defuse(fusedRange, term.indices, I)


  def generate(self, cpp, routineCache, gemm_cfg):
    """
    Args:
      cpp (IO): a file stream
      routineCache (RoutineCache):
      gemm_cfg (GeneratorCollection):

    Returns:

    """
    descr = self._descr


    A = descr.leftTerm.indices - descr.loopIndices
    B = descr.rightTerm.indices - descr.loopIndices
    C = descr.result.indices - descr.loopIndices
    Im = set(A) & set(C)
    In = set(B) & set(C)
    Ik = set(A) & set(B)


    hasOuterLoops = len(descr.outerLoopIndices) > 0
    outerAname = '_A' if hasOuterLoops else descr.leftTerm.name
    outerBname = '_B' if hasOuterLoops else descr.rightTerm.name
    outerCname = '_C' if hasOuterLoops else descr.result.name
    outerPrefetchName = '_Cprefetch' if hasOuterLoops and descr.prefetchName is not None else descr.prefetchName


    hasInnerLoops = len(descr.innerLoopIndices) > 0
    innerAname = '_Ain' if hasInnerLoops else outerAname
    innerBname = '_Bin' if hasInnerLoops else outerBname
    innerCname = '_Cin' if hasInnerLoops else outerCname
    innerPrefetchName = '_Cprefetchin' if hasInnerLoops and outerPrefetchName is not None else outerPrefetchName

    alignedStartA = not hasOuterLoops or self._alignedStart(descr.leftTerm, descr.outerLoopIndices)

    AmemLayout = self._memLayout(descr.leftTerm, Im, Ik)
    BmemLayout = self._memLayout(descr.rightTerm, Ik, In)
    CmemLayout = self._memLayout(descr.result, Im, In)

    Aeqspp = self._reduce(descr.leftTerm, A, AmemLayout)
    Beqspp = self._reduce(descr.rightTerm, B, BmemLayout)
    Ceqspp = self._reduce(descr.result, C, CmemLayout)

    gemmDescr = gemm.Description(
      leftTerm=TensorDescription(innerAname, AmemLayout, Aeqspp),
      rightTerm=TensorDescription(innerBname, BmemLayout, Beqspp),
      result=TensorDescription(innerCname, CmemLayout, Ceqspp),
      transA=descr.transA,
      transB=descr.transB,
      alpha=descr.alpha,
      beta = 1.0 if descr.add else 0.0,
      arch=self._arch,
      alignedStartA=self._alignedStart(descr.leftTerm, descr.outerLoopIndices) and self._alignedStart(descr.leftTerm, descr.innerLoopIndices),
      alignedStartC=self._alignedStart(descr.result, descr.outerLoopIndices) and self._alignedStart(descr.result, descr.innerLoopIndices),
      prefetchName=innerPrefetchName
    )

    if not descr.add:
      lr = dict()
      m, n, k = gemmDescr.mnk()
      lr.update(descr.loopRanges)
      lr.update(self._defuse(m, descr.leftTerm, Im))
      lr.update(self._defuse(n, descr.rightTerm, In))
      writeBB = boundingBoxFromLoopRanges(descr.result.indices, lr)
      initializeWithZero(cpp, self._arch, descr.result, writeBB)


    class LoGBody(object):
      def __call__(s):
        if hasInnerLoops:

          self._pointer(cpp=cpp,
                        targetName=innerAname,
                        baseName=outerAname,
                        term=descr.leftTerm,
                        loopIndices=descr.innerLoopIndices)

          self._pointer(cpp=cpp,
                        targetName=innerBname,
                        baseName=outerBname,
                        term=descr.rightTerm,
                        loopIndices=descr.innerLoopIndices)

          self._pointer(cpp=cpp,
                        targetName=innerCname,
                        baseName=outerCname,
                        term=descr.result,
                        loopIndices=descr.innerLoopIndices,
                        const=False)

          if outerPrefetchName is not None:
            self._pointer(cpp=cpp,
                          targetName=innerPrefetchName,
                          baseName=outerPrefetchName,
                          term=descr.result,
                          loopIndices=descr.innerLoopIndices)

        if descr.is_cuda_factory_used:
          # NOTE: jump table is needed for GPU based GEMM generator
          jump_table = self._generate_tensors_jumps(cpp=cpp,
                                                    tensor_descriptions=descr)
        else:
          jump_table = None


        generator = gemm.generator(self._arch, gemmDescr, gemm_cfg)
        return generator.generate(cpp, routineCache, additional=jump_table)


    class InnerLoopBody(object):
      def __call__(s):
        flops = 0
        if hasOuterLoops:
          self._pointer(cpp=cpp,
                        targetName=outerAname,
                        baseName=descr.leftTerm.name,
                        term=descr.leftTerm,
                        loopIndices=descr.outerLoopIndices)

          self._pointer(cpp=cpp,
                        targetName=outerBname,
                        baseName=descr.rightTerm.name,
                        term=descr.rightTerm,
                        loopIndices=descr.outerLoopIndices)

          self._pointer(cpp=cpp,
                        targetName=outerCname,
                        baseName=descr.result.name,
                        term=descr.result,
                        loopIndices=descr.outerLoopIndices,
                        const=False)

          if descr.prefetchName is not None:
            self._pointer(cpp=cpp,
                          targetName=outerPrefetchName,
                          baseName=descr.prefetchName,
                          term=descr.result,
                          loopIndices=descr.outerLoopIndices)

        """
        self._generate_tensors_jumps(cpp=cpp,
                                    target_name_A=outerAname,
                                    target_name_B=outerBname,
                                    target_name_C=outerCname,
                                    tensor_descriptions=descr)

        """

        #TODO: find another way to surround a block of code with curly brackets
        cpp('{}'.format('{' if descr.is_cuda_factory_used else ''))
        if descr.assignLoopRanges is not None:
          gemmDescr.setBeta(0.0)
          flops += forLoops(cpp=cpp,
                            indexNames=descr.innerLoopIndices,
                            ranges=descr.assignLoopRanges,
                            body=LoGBody(),
                            pragmaSimd=False)

        if descr.addLoopRanges is not None:
          gemmDescr.setBeta(1.0)
          flops += forLoops(cpp=cpp,
                            indexNames=descr.innerLoopIndices,
                            ranges=descr.addLoopRanges,
                            body=LoGBody(),
                            pragmaSimd=False)
        cpp('{}'.format('}' if descr.is_cuda_factory_used else ''))

        return flops

    return forLoops(cpp=cpp,
                    indexNames=descr.outerLoopIndices,
                    ranges=descr.loopRanges,
                    body=InnerLoopBody(),
                    pragmaSimd=False)

