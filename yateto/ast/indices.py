import sys
import functools
from .. import aspp


class Indices(object):
  def __init__(self, indexNames: str = '', shape: tuple = ()):
    """TODO: complete description

    Args:
      indexNames: each character of a string defines a name of the corresponding index
      shape: shape of the tensor i.e. sizes of each dimension
    """

    self._indices = tuple(indexNames)
    self._size = dict()
    
    assert len(self._indices) == len(set(self._indices)), 'Repeated indices are not allowed ({}).'.format(indexNames)
    assert len(self._indices) == len(shape), 'Indices {} do not match tensor shape {}.'.format(str(self), shape)

    # create a table where the key of each entry is a character of
    # the corresponding index and the value is a size of the dimension
    # defined by the index i.e. a simple lookup table which allows to
    # find the dimension of an index by its name
    self._size = {self._indices[i]: size for i, size in enumerate(shape)}
  
  def tostring(self):
    return ''.join(self._indices)
  
  def extract(self, indexNames):
    return Indices(str(indexNames), self.subShape(indexNames))
  
  def firstIndex(self):
    return self.extract(self._indices[0])

  def shape(self):
    return self.subShape(self._indices)
  
  def subShape(self, indexNames):
    return tuple([self._size[index] for index in indexNames])

  def indexSize(self, index):
    return self._size[index]
  
  def permuted(self, indexNames):
    assert set(indexNames) == set(self)
    return Indices(indexNames, self.subShape(indexNames))
    
  def find(self, index):
    assert len(index) == 1
    return self._indices.index(index)
  
  def positions(self, I, sort=True):
    pos = [self.find(i) for i in I]
    if sort:
      return sorted(pos)
    return pos
  
  def __eq__(self, other):
    return other != None and self._indices == other._indices and self._size == other._size
    
  def __ne__(self, other):
    return other == None or self._indices != other._indices or self._size != other._size
  
  def __hash__(self):
    return hash((self._indices, self.shape()))
  
  def __iter__(self):
    return iter(self._indices)
  
  def __getitem__(self, key):
    return self._indices[key]
    
  def __len__(self):
    return len(self._indices)
  
  def __and__(self, other):
    return set(self) & set(other)
  
  def __rand__(self, other):
    return self & other
    
  def __le__(self, other):
    indexNamesContained = set(self._indices) <= set(other._indices)
    return indexNamesContained and all([self._size[index] == other._size[index] for index in self._indices])


  def __sub__(self, other):
    """Exclude indices from an instance which match to any character of the provided string object.

    The method returns a new instance of a class.

    Args:
      other (str): a string of indices

    Returns:
      Indices: a new instance if Indices class with exclude indices

    Examples:
      >>> from yateto.ast.indices import Indices
      >>> shape = (3,2,4)
      >>> index_names = 'ijk'
      >>> obj = Indices(index_names, shape)
      >>> other_indices = 'ik'
      >>> obj - other_indices
      (j=2)
      >>> (obj - other_indices)._indices
      ('j',)
    """
    indexNames = [index for index in self._indices if index not in other]
    return Indices(indexNames, self.subShape(indexNames))


  def merged(self, other):
    indexNames = self._indices + other._indices
    shape = self.subShape(self._indices) + other.subShape(other._indices)
    return Indices(indexNames, shape)
    
  def sorted(self):
    indexNames = sorted(self._indices)
    return Indices(indexNames, self.subShape(indexNames))
  
  def __str__(self):
    return self.tostring()
    
  def __repr__(self):
    return '({})'.format(','.join(['{}={}'.format(index, self._size[index]) for index in self._indices]))
  
  def size(self):
    return self._size


class Range(object):
  def __init__(self, start, stop):
    """
    Args:
      start (int): min value of a given tensor index range
      stop (int): max value of a given tensor index range
    """
    self.start = start
    self.stop = stop
  
  def size(self):
    """
    Returns:
      int: the size of a range i.e. a size of a tensor index range
    """
    return self.stop - self.start


  def aligned(self, arch):
    """Adjusts a indices of a range in a way suitable for vectorization

    Args:
      arch (Architecture): a specific target compute architecture which the source code is going
                           to be generated for

    Returns:
      Range: a new instance of Range class with aligned indices suitable for vectorization of
             a given compute architecture

    Examples:
      >>> from yateto.arch import Architecture
      >>> arch = Architecture(name='hsw', precision='D', alignment=32)
      >>> from yateto.ast.indices import Range
      >>> range = Range(start=3, stop=13)
      >>> range.aligned(arch)
      <yateto.ast.indices.Range object at 0x7f44c92003c8>
      >>> str(range.aligned(arch))
      'Range(0, 16)'
    """
    return Range(arch.alignedLower(self.start), arch.alignedUpper(self.stop))


  def __and__(self, other):
    """TODO

    Args:
      other (Range): TODO

    Returns:
      Range: TODO

    Examples:
      >>> from yateto.ast.indices import Range
      >>> range_1 = Range(0, 5)
      >>> range_2 = Range(3, 9)
      >>> str(range_1 & range_2)
      'Range(3, 5)'
    """
    return Range(max(self.start, other.start), min(self.stop, other.stop))


  def __or__(self, other):
    """TODO

    Args:
      other (Range): TODO

    Returns:
      Range: TODO

    Examples:
      >>> from yateto.ast.indices import Range
      >>> range_1 = Range(0, 5)
      >>> range_2 = Range(3, 9)
      >>> str(range_1 | range_2)
      'Range(0, 9)'
    """
    return Range(min(self.start, other.start), max(self.stop, other.stop))


  def __contains__(self, other):
    """Checks whether a provided range is inside

    Args:
      other (Range): a provided range

    Returns:
      bool: True, if a provided range is inside of the current instance of Range class. \
            Otherwise, False.

    Examples:
      >>> from yateto.ast.indices import Range
      >>> range_1 = Range(0, 10)
      >>> range_2 = Range(3, 7)
      >>> range_2 in range_1
      True

      >>> from yateto.ast.indices import Range
      >>> range_1 = Range(0, 10)
      >>> range_2 = Range(3, 11)
      >>> range_2 in range_1
      False
    """
    return self.start <= other.start and self.stop >= other.stop


  def __eq__(self, other):
    """
    Args:
      other (Range): a comparable instance of Range

    Returns:
      bool: True, if both max and min values of two ranges coincide. Otherwise, False
    """
    return self.start == other.start and self.stop == other.stop


  def __str__(self):
    return 'Range({}, {})'.format(self.start, self.stop)
  
  def __iter__(self):
    """Allows to consequently iterate between min and max values of a range

    Returns:
      range_iterator: an iterator instance
    """
    return iter(range(self.start, self.stop))


class BoundingBox(object):
  def __init__(self, listOfRanges):
    """
    Args:
      listOfRanges (List[Range]):
    """
    self._box = listOfRanges


  @classmethod
  def fromSpp(cls, spp):
    """Creates an instance of a BoundingBox class given a sparsity pattern of a tensor

    Args:
      spp (Type[ASpp]): a sparsity pattern

    Returns:
      BoundingBox: an instance of a BoundingBox class
    """
    return cls([Range(min, max + 1) for min, max in spp.nnzbounds()])


  def size(self):
    """
    Returns:
      int: size of a bounding box (i.e. a tensor volume)
    """
    total_size = 1
    for range in self._box:
      total_size *= range.size()
    return total_size


  def __contains__(self, entry):
    if len(entry) != len(self):
      return False

    if len(self) == 0:
      return True

    if isinstance(entry[0], Range):
      return all([e in self[i] for i, e in enumerate(entry)])

    return all([e >= self[i].start and e <= self[i].stop for i, e in enumerate(entry)])


  def __getitem__(self, key):
    """
    Args:
      key (int): an index of a dimension

    Returns:
      Range: the corresponding range of a given dimension
    """
    return self._box[key]


  def __len__(self):
    """
    Returns:
      int: number of ranges (i.e. number of dimensions)
    """
    return len(self._box)


  def __iter__(self):
    """Allows to consequently iterate between all ranges inside of a bounding box

    Returns:
      range_iterator: an iterator instance
    """
    return iter(self._box)


  def __eq__(self, other):
    """Iterates through all ranges of two bounding boxes and checks whether the corresponding
    ranges are the same.

    Args:
      other (BoundingBox): a comparable instance of BoundingBox

    Returns:
      bool: if boxes have the same ranges. Otherwise, False
    """
    return all([s == o for s, o in zip(self, other)])


  def __str__(self):
    return '{}({})'.format(type(self).__name__, ', '.join([str(r) for r in self]))


@functools.total_ordering
class LoGCost(object):    
  def __init__(self,
               stride=sys.maxsize,
               leftTranspose=sys.maxsize,
               rightTranspose=sys.maxsize,
               fusedIndices=0):
    """
    stride (w.r.t. first dimension): 0 = unit stride, 1 non-unit stride (lower is better)
    transpose: Number of required transposes                            (lower is better)
    fusedIndices: Number of tensor indices to be fused in a super-index (higher is better)
    """
    self._stride = stride
    self._leftTranspose = leftTranspose
    self._rightTranspose = rightTranspose
    self._fusedIndices = fusedIndices
  
  @staticmethod
  def addIdentity():
    return LoGCost(0, 0, 0, 0)
    
  def _totuple(self):
    # minus sign before _fusedIndices as higher is better
    return (self._stride, self._leftTranspose + self._rightTranspose, -self._fusedIndices)
  
  def __lt__(self, other):
    s = self._totuple()
    o = other._totuple()
    if s == o:
      return self._leftTranspose < other._leftTranspose
    return self._totuple() < other._totuple()

  def __eq__(self, other):
    return self._totuple() == other._totuple() and self._leftTranspose == other._leftTranspose
  
  def __add__(self, other):
    return LoGCost(self._stride + other._stride,
                   self._leftTranspose + other._leftTranspose,
                   self._rightTranspose + other._rightTranspose,
                   self._fusedIndices + other._fusedIndices)
  
  def __repr__(self):
    return '{{stride: {}, left transpose: {}, right transpose: {}, fused indices: {}}}'.format(self._stride, self._leftTranspose, self._rightTranspose, self._fusedIndices)
