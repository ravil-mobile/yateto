from .graph import *
from collections import deque


class MergeScalarMultiplications(object):   
  def visit(self, cfg):
    n = len(cfg)-1
    i = 1
    while i < n:
      ua = cfg[i].action
      if ua.isRHSVariable() and not ua.isCompound() and ua.scalar is not None:
        va = cfg[i-1].action
        if va.isRHSExpression() and not va.isCompound() and ua.term == va.result:
          va.scalar = ua.scalar
          va.result = ua.result
          del cfg[i]
          i -= 1
          n -= 1
      i += 1
    return cfg


class LivenessAnalysis(object):
  def visit(self, cfg):
    cfg[-1].live = set()
    for i in reversed(range(len(cfg)-1)):
      cfg[i].live = (cfg[i+1].live - {cfg[i].action.result}) | cfg[i].action.variables()
    return cfg


class SubstituteForward(object):
  def visit(self, cfg):
    n = len(cfg)-1
    for i in range(n):
      ua = cfg[i].action
      v = cfg[i+1]
      if not ua.isCompound() and ua.isRHSVariable() and ua.term.writable and ua.result.isLocal() and ua.term not in v.live:
        when = ua.result
        by = ua.term
        maySubs = all([cfg[j].action.maySubstitute(when, by) for j in range(i, n)])
        if maySubs:
          for j in range(i, n):
            cfg[j].action = cfg[j].action.substituted(when, by)
          cfg = LivenessAnalysis().visit(cfg)
    return cfg


class SubstituteBackward(object):
  def visit(self, cfg):
    n = len(cfg)-1
    for i in reversed(range(n)):
      va = cfg[i].action
      if not va.isCompound() and va.isRHSVariable() and va.term.isLocal():
        by = va.result
        found = -1
        for j in range(i):
          u = cfg[j]
          if by not in u.live and not u.action.isCompound() and u.action.result == va.term:
            found = j
            break
        if found >= 0:
          when = u.action.result
          maySubs = cfg[found].action.maySubstitute(when, by, term=False) and all([cfg[j].action.maySubstitute(when, by) for j in range(found+1,i+1)])
          if maySubs:
            cfg[found].action = cfg[found].action.substituted(when, by, term=False)
            for j in range(found+1,i+1):
              cfg[j].action = cfg[j].action.substituted(when, by)
            cfg = LivenessAnalysis().visit(cfg)
    return cfg


class RemoveEmptyStatements(object):
  def visit(self, cfg):
    n = len(cfg)-1
    i = 0
    while i < n:
      ua = cfg[i].action
      if not ua.isCompound() and ua.isRHSVariable() and ua.result == ua.term and ua.hasTrivialScalar():
        del cfg[i]
        n -= 1
      else:
        i += 1
    return cfg


class MergeActions(object):
  def visit(self, cfg):
    n = len(cfg)-1
    i = 0
    while i < n:
      ua = cfg[i].action
      if not ua.isCompound():
        found = -1
        V = ua.variables()
        for j in range(i+1,n):
          va = cfg[j].action
          if va.isRHSVariable() and ua.result == va.term and va.result not in V and (ua.hasTrivialScalar() or va.hasTrivialScalar()):
            found = j
            break
          elif ua.result in va.variables() or ua.result == va.result:
            break
          else:
            V = V | va.variables() | {va.result}
        if found >= 0:
          va = cfg[found].action
          if ua.maySubstitute(ua.result, va.result, term=False):
            cfg[i].action = ua.substituted(ua.result, va.result, term=False)
            cfg[i].action.add = va.add
            if not va.hasTrivialScalar():
              cfg[i].action.scalar = va.scalar
            del cfg[found]
            n -= 1
      i += 1
    return LivenessAnalysis().visit(cfg)


class DetermineLocalInitialization(object):
  def visit(self, cfg):
    """TODO

    .. uml:: _static/yateto/controlflow/transormer/DetermineLocalInitialization.visit.uml

    Args:
      cfg (List[ProgramPoint]): an execution block (a control flow graph)

    Returns:
      List[ProgramPoint]: an augmented execution block where each program point knows \
                          which buffer index a result of an expression should use
    """

    lcls = dict()
    num_buffers_counter = 0  # int
    used_buffers_table = dict()  # Dict[Variable, int]
    free_buffers_deque = deque()  # Deque[int]
    buffer_size_table = dict()  # Dict[int, int]

    for program_point in cfg:
      # a table of buffers that have to be initialized before a program point execution
      program_point.initBuffer = dict()  # Dict[int, int]

      # TODO
      program_point.bufferMap = dict()

    # iterate through each program point
    n = len(cfg)
    for i in range(n-1):

      # tale a program action from a program point
      program_action = cfg[i].action


      if program_action and not program_action.isCompound() and program_action.result.isLocal():
        # at this point we consider only expressions
        # which have a form y += x i.e. (y = y + x) and
        # the results are held in temporary varaibles

        if len(free_buffers_deque) > 0:
          # assign a value to a current buffer index from the top
          # of free buffers deque if free buffers deque is not empty
          current_buffer_index = free_buffers_deque.pop()
        else:
          # Generate a new buffer index if free_buffers_deque is empty
          current_buffer_index = num_buffers_counter
          num_buffers_counter += 1


        # tell a program point i that the right hand side of the corresponding expression
        # has to use a buffer denoted with current_buffer_index
        cfg[i].bufferMap[program_action.result] = current_buffer_index

        # dublicate the above information to the used buffers table
        used_buffers_table[program_action.result] = current_buffer_index


        # compute memory size which is needed to hold a result of an expression in reals
        needed_buffer_size = program_action.result.memoryLayout().requiredReals()


        if current_buffer_index in buffer_size_table:
          privious_buffer_size = buffer_size_table[current_buffer_index],
          buffer_size_table[current_buffer_index] = max(privious_buffer_size,
                                                        needed_buffer_size)
        else:
          buffer_size_table[current_buffer_index] = needed_buffer_size


      # compute a set of variables which are not going to be used
      # at the next program point i.e. local variables for the current program point
      free = cfg[i].live - cfg[i+1].live

      # iterate through all varaibles that are not going to be used at the next program point
      for local in free:

        # Check whether a variable used a buffer
        if local in used_buffers_table:

          # Remove a varaible from the used buffer table and put it at the top
          # of free buffer deque
          free_buffers_deque.appendleft(used_buffers_table.pop(local))

    if len(cfg) > 0:
      # the first program point is going to know how many buffers are needed
      # and which memory size each of them requires
      cfg[0].initBuffer = buffer_size_table

    return cfg
