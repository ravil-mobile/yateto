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
  flops = 0
  if indexNo == None:
    indexNo = len(indexNames)-1

  if indexNo < 0:
    flops = body()
  else:
    index = indexNames[indexNo]
    rng = ranges[index]

    if pragmaSimd and indexNo == 0:
      cpp('#pragma omp simd')

    with cpp.For('int {0} = {1}; {0} < {2}; ++{0}'.format(index, rng.start, rng.stop)):
      flops = forLoops(cpp, indexNames, ranges, body, pragmaSimd, indexNo-1)
    flops = flops * rng.size()

  return flops


def loopRanges(term: IndexedTensorDescription, loopIndices):
  overlap = set(loopIndices) & set(term.indices)
  bbox = BoundingBox.fromSpp(term.eqspp)
  return {index: bbox[term.indices.find(index)] for index in overlap}


def testLoopRangesEqual(A, B):
  overlap = A.keys() & B.keys()
  return all([A[index] == B[index] for index in overlap])


def testLoopRangesAContainedInB(A, B):
  overlap = A.keys() & B.keys()
  return all([A[index] in B[index] for index in overlap])


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
  """ TODO: write some generated to a file

  Args:
    cpp: a file descriptor
    arch (Architecture): a description of the target compute architecture
    result (IndexTensorDescription): TODO
    writeBB (BoundingBox): a region of a tensor (memory layout) which the user wants to write
                           data in
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
        min_region_address, max_region_address = min(region), max(region)
        initial_address = '{} + {}'.format(result.name, min_region_address)
        cpp.memset(initial_address, max_region_address - min_region_address + 1, arch.typename)

  else:
    cpp.memset(result.name, result.memoryLayout.requiredReals(), arch.typename)
