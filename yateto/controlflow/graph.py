from yateto.type import Scalar
from yateto.ast.node import Node
from typing import Type, Union


class Variable(object):
  def __init__(self, name, writable, memoryLayout, eqspp=None, tensor=None):
    self.name = name
    self.writable = writable
    self.tensor = tensor
    self._memoryLayout = memoryLayout
    self._eqspp = eqspp

  def variables(self):
    return {self}

  def maySubstitute(self, when, by):
    return self.substituted(when, by).memoryLayout().isCompatible(self.eqspp())

  def substituted(self, when, by, memoryLayout=None):
    return by if self == when else self

  def resultCompatible(self, result):
    return result.memoryLayout().isCompatible(self.eqspp())

  def isGlobal(self):
    return self.tensor is not None

  def isLocal(self):
    return not self.isGlobal()

  def memoryLayout(self):
    return self._memoryLayout

  def eqspp(self):
    return self._eqspp

  def __hash__(self):
    return hash(self.name)

  def __str__(self):
    return self.name

  def __repr__(self):
    return str(self)

  def __eq__(self, other):
    isEq = self.name == other.name
    assert not isEq or (self.writable == other.writable
                        and self._memoryLayout == other._memoryLayout)
    return isEq


class Expression(object):
  def __init__(self, node, memoryLayout, variables):
    self.node = node
    self._memoryLayout = memoryLayout
    self._variables = variables

  def memoryLayout(self):
    return self._memoryLayout

  def eqspp(self):
    return self.node.eqspp()

  def variables(self):
    return set([var for var in self._variables])

  def variableList(self):
    return self._variables

  def maySubstitute(self, when, by):
    layouts = [var.substituted(when, by).memoryLayout() for var in self._variables]
    c1 = all(layouts[i].isCompatible(var.eqspp()) for i, var in enumerate(self._variables))
    c2 = self.node.argumentsCompatible(layouts)
    return c1 and c2

  def substituted(self, when, by, memoryLayout):
    return Expression(self.node, memoryLayout, [var.substituted(when, by)
                                                for var in self._variables])

  def resultCompatible(self, result):
    c1 = result.memoryLayout().isCompatible(self.eqspp())
    c2 = self.node.resultCompatible(result.memoryLayout())
    return c1 and c2

  def __str__(self):
    return '{}({})'.format(type(self.node).__name__, ', '.join([str(var)
                                                                for var in self._variables]))

class ProgramAction(object):
  def __init__(self,
               result: Variable,
               term: Variable,
               add: bool,
               scalar: Union[float, Scalar] = None):
    """TODO: Complete description.

    Args:
      result:
      term:
      add: a flag which tells whether += or = sign is used for a program action
      scalar:
    """

    self.result = result
    self.term = term
    self.add = add
    self.scalar = scalar

  def isRHSExpression(self):
    return isinstance(self.term, Expression)

  def isRHSVariable(self):
    return not self.isRHSExpression()

  def isCompound(self):
    return self.add

  def hasTrivialScalar(self):
    return self.scalar is None or self.scalar == 1.0

  def variables(self):
    V = self.term.variables()
    if self.add:
      V = V | self.result.variables()
    return V

  def maySubstitute(self, when, by, result=True, term=True):
    maySubsTerm = self.term.maySubstitute(when, by)
    maySubsResult = self.result.maySubstitute(when, by)

    rsubs = self.result.substituted(when, by) if result else self.result
    tsubs = self.term.substituted(when, by, rsubs.memoryLayout()) if term else self.term

    compatible = tsubs.resultCompatible(rsubs)

    return (not term or maySubsTerm) and (not result or maySubsResult) and compatible

  def substituted(self, when, by, result=True, term=True):
    rsubs = self.result.substituted(when, by) if result else self.result
    tsubs = self.term.substituted(when, by, rsubs.memoryLayout()) if term else self.term
    return ProgramAction(rsubs, tsubs, self.add, self.scalar)


class ProgramPoint(object):
  def __init__(self, action: ProgramAction):
    """TODO: complete description

    Args:
      action:
    """

    self.action = action
    self.live = None
    self.initBuffer = None
    self.bufferMap = None
