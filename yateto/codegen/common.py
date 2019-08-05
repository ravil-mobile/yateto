from .. import aspp
from ..ast.indices import BoundingBox
from ..ast.log import splitByDistance

class TensorDescription(object):
  def __init__(self, name, memoryLayout, eqspp):
    self.name = name
    self.memoryLayout = memoryLayout
    self.eqspp = eqspp
    BoundingBox(eqspp)
  
  @classmethod
  def fromNode(cls, name, node):
    return cls(name, node.memoryLayout(), node.eqspp())


class IndexedTensorDescription(TensorDescription):
  def __init__(self, name, indices, memoryLayout, eqspp):
    """
    Args:
      name (str): a variable name
      indices (TODO):
      memoryLayout (TODO):
      eqspp (TODO):
    """
    super().__init__(name, memoryLayout, eqspp)
    self.indices = indices


  @classmethod
  def fromNode(cls, var, node):
    """Creates a tensor description from a variable and a node.

    Args:
      var (Variable):
      node (Type[Node]):

    Returns:
      IndexedTensorDescription: TODO
    """
    return cls(str(var), node.indices, var.memoryLayout(), node.eqspp())


def forLoops(cpp, indexNames, ranges, body, pragmaSimd=True, indexNo=None):
  """Generates code for a tensor operation using nested for-loops

  NOTE: parameter 'indexNo' is usually used by the function itself during recursive calls.
  The user calls the function with 'indexNo' equal to None.

  Args:
    cpp (IO): a file stream
    indexNames: TODO
    ranges: TODO
    body (Callable[[],[]]): a callable object which knows how to construct the inner body
                            of a loop
    pragmaSimd (bool): a flag whether to include SIMD pragma to the source code
    indexNo (int): TODO

  Returns:
    int: the number of floating point operations
  """

  flops = 0

  # adjust 'indexNo' counter. It is needed during
  # the first function call made by the user
  if indexNo == None:
    indexNo = len(indexNames) - 1


  if indexNo < 0:

    # stop recursion and generate inner loop body
    flops = body()
  else:

    # generate a next for-loop statement
    index = indexNames[indexNo]
    rng = ranges[index]

    # insert pragma if it is needed
    if pragmaSimd and indexNo == 0:
      cpp('#pragma omp simd')

    with cpp.For('int {0} = {1}; {0} < {2}; ++{0}'.format(index, rng.start, rng.stop)):

      # recursively call the next for-loop statement
      flops = forLoops(cpp, indexNames, ranges, body, pragmaSimd, indexNo - 1)

    # adjust the number of flops
    flops = flops * rng.size()

  return flops


def loopRanges(term, loopIndices):
  """Computes a table where a key is an index of a tensor description and the corresponding value
  is a range of this index (see, class Range)

  Args:
    term (IndexedTensorDescription): a description of a variable (controlflow) of an execution
                                     block (aka cfg)
    loopIndices (Indices): other indices

  Returns:
    Dict[str, Range]: a table of ranges
  """

  # compute common indices
  overlap = set(loopIndices) & set(term.indices)

  # create a bounding box from a tensor description
  bbox = BoundingBox.fromSpp(term.eqspp)

  # create and return a table
  # NOTE: bbox[<int>] returns an instance of class Range
  return {index_name: bbox[term.indices.find(index_name)] for index_name in overlap}


def testLoopRangesEqual(A, B):
  overlap = A.keys() & B.keys()
  return all([A[index] == B[index] for index in overlap])


def testLoopRangesAContainedInB(A, B):
  """Checks inclusion of tensor B in A

  Args:
    A (Dict[str, Range]): a table of ranges of tensor A
    B (Dict[str, Range]): a table of ranges of tensor B

  Returns:
    bool: True, if all ranges of tensor B are inside of ranges of tensor A
  """
  overlap = A.keys() & B.keys()
  return all([A[index_names] in B[index_names] for index_names in overlap])


def boundingBoxFromLoopRanges(indices, loopRanges):
  """Generate an instance of BoundingBox class with given indices names and
  a table of ranges.

  NOTE: it is a factory function

  Args:
    indices (Tuple[str]): tensor indices
    loopRanges (Dict[str, Range]): a table of tensor ranges

  Returns:
    BoundingBox: a new instance of BoundingBox class
  """
  return BoundingBox([loopRanges[index] for index in indices])


def reduceSpp(spp, sourceIndices, targetIndices):
  return spp.indexSum(sourceIndices, targetIndices)


def initializeWithZero(cpp, arch, result: TensorDescription, writeBB=None):
  """ TODO:

  Args:
    cpp: a file stream
    arch (Architecture): a description of the target compute architecture
    result (IndexTensorDescription): a detailed description of a tensor
    writeBB (Union[bool, BoundingBox): a region of a tensor (memory layout) which
                                      the user wants to write data in
  """

  if writeBB:

    # compute and sort all elements of a tensor outside the user's provided
    # writable bounding box (writeBB)
    addresses = sorted(result.memoryLayout.notWrittenAddresses(writeBB))

    if len(addresses) > 0:

      # split the computed address into contiguous regions of address
      regions = splitByDistance(addresses)

      # iterate through all regions and generate code to initialize
      # these regions with zeros
      for region in regions:

        # compute min and max address of a contiguous range
        min_region_address, max_region_address = min(region), max(region)

        # generate code
        initial_address = '{} + {}'.format(result.name, min_region_address)
        cpp.memset(initial_address, max_region_address - min_region_address + 1, arch.typename)

  else:
    cpp.memset(result.name, result.memoryLayout.requiredReals(), arch.typename)
