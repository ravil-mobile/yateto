from ..ast.node import IndexedTensor
from .common import TensorDescription, IndexedTensorDescription
from . import copyscaleadd, indexsum, log, product

class Factory(object):
  def create(self, node, *args):
    method = 'create_' + node.__class__.__name__
    factory = getattr(self, method, self.generic_create)
    return factory(node, *args)
  
  def generic_create(self, node, *args):
    raise NotImplementedError

class KernelFactory(Factory):
  def __init__(self, cpp, arch):
    self._cpp = cpp
    self._arch = arch

  def create_LoopOverGEMM(self, node, resultName, argNames, add):
    assert len(argNames) == 2
    description = log.Description(
      add = add,
      result = IndexedTensorDescription.fromNode(resultName, node),
      leftTerm = IndexedTensorDescription.fromNode(argNames[0], node.leftTerm()),
      rightTerm = IndexedTensorDescription.fromNode(argNames[1], node.rightTerm()),
      loopIndices = node.loopIndices(),
      transA = node.transA(),
      transB = node.transB()
    )
    generator = log.generator(self._arch, description)
    generator.generate(self._cpp)
  
  def create_IndexSum(self, node, resultName, argNames, add):
    assert len(argNames) == 1
    description = indexsum.Description(
      add = add,
      result = IndexedTensorDescription.fromNode(resultName, node),
      term = IndexedTensorDescription.fromNode(argNames[0], node.term())
    )
    generator = indexsum.generator(self._arch, description)
    generator.generate(self._cpp)
  
  def create_Product(self, node, resultName, argNames, add):
    assert len(argNames) == 2
    description = product.Description(
      add = add,
      result = IndexedTensorDescription.fromNode(resultName, node),
      leftTerm = IndexedTensorDescription.fromNode(argNames[0], node.leftTerm()),
      rightTerm = IndexedTensorDescription.fromNode(argNames[1], node.rightTerm())
    )
    generator = product.generator(self._arch, description)
    generator.generate(self._cpp)
  
  def create_Add(self, node, resultName, argNames, add):
    beta = 1.0 if add else 0.0
    for i,child in enumerate(node):
      if isinstance(child, IndexedTensor):
        description = copyscaleadd.Description(
          alpha = 1.0,
          beta = beta,
          result = TensorDescription.fromNode(resultName, node),
          term = TensorDescription.fromNode(argNames[i], child),
        )
        generator = copyscaleadd.generator(self._arch, description)
        generator.generate(self._cpp)
      beta = 1.0
