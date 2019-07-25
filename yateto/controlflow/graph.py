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
    """

    Returns:
      Set[Variable]: a singleton which consists of the instance itself
    """
    return {self}


  def maySubstitute(self, when, by):
    return self.substituted(when, by).memoryLayout().isCompatible(self.eqspp())

  def substituted(self, when, by, memoryLayout=None):
    return by if self == when else self

  def resultCompatible(self, result):
    return result.memoryLayout().isCompatible(self.eqspp())


  def isGlobal(self):
    """

    Returns:
      bool: True, if a variable corresponds to a tensor. Otherwise, False
    """
    return self.tensor is not None


  def isLocal(self):
    """

    Returns:
      bool: True, if a variable is used as a temporary one. Otherwise, False
    """
    return not self.isGlobal()


  def memoryLayout(self):
    return self._memoryLayout


  def eqspp(self):
    return self._eqspp


  def __hash__(self):
    return hash(self.name)


  def __str__(self):
    """
    Returns:
      str: a name of a variable
    """
    return self.name


  def __repr__(self):
    """
    Returns:
      str: a name of a variable
    """
    return str(self)


  def __eq__(self, other):
    isEq = self.name == other.name
    assert not isEq or (self.writable == other.writable
                        and self._memoryLayout == other._memoryLayout)
    return isEq


class Expression(object):
  def __init__(self, node, memoryLayout, variables):
    """
    Args:
      node (Type[Node]):
      memoryLayout:
      variables (List[Variable]):
    """
    self.node = node
    self._memoryLayout = memoryLayout
    self._variables = variables

  def memoryLayout(self):
    return self._memoryLayout

  def eqspp(self):
    return self.node.eqspp()


  def variables(self):
    """
    Returns:
      Set[Variable]: a set of variables involved into an expression
    """
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
      result: TODO
      term: TODO
      add: a flag which tells whether += or = sign is used for a program action
      scalar: TODO
    """

    self.result = result
    self.term = term
    self.add = add
    self.scalar = scalar


  def isRHSExpression(self):
    """Tells whether the right hand side is an expression or not i.e. if the right hand side
    assumes some computations

    Returns:
      bool: True, if the right hand side requres some computation to be performed. \
            Otherwise, False
    """
    return isinstance(self.term, Expression)


  def isRHSVariable(self):
    """Tells whether the right hand side is a variable or not

    NOTE: the method just negates a result of 'isRHSExpression' method call.

    Returns:
      bool: True, if the right-hand side of an expresion is just a variable i.e. the current \
            expression is just an re-assignment
    """
    return not self.isRHSExpression()


  def isCompound(self):
    """Tells whether a program point has a form: y += x i.e. y = y + x or not

    Returns:
      bool: True, if an expression is compound. False, otherwise
    """
    return self.add


  def hasTrivialScalar(self):
    """

    Returns:
      bool: True, if a constant involved into an expression doesn't exist or equal to 1. \
            Otherwise, False
    """
    return self.scalar is None or self.scalar == 1.0


  def variables(self):
    """Returns all variables involved in the right-hand side.

    NOTE: A program point can represent either a = b or a = a + b expression

    Returns:
      Set[Variable]: variables of the right-hand side
    """
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
