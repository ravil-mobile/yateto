from .ast.indices import BoundingBox, Range
import copy
import itertools
import warnings
import numpy as np
from abc import ABC, abstractmethod


class MemoryLayout(ABC):
  def __init__(self, shape):
    self._shape = shape

  def shape(self):
    return self._shape
  
  @abstractmethod
  def address(self, entry):
    pass
  
  @abstractmethod
  def subtensorOffset(self, topLeftEntry):
    pass

  @abstractmethod
  def alignedStride(self):
    return False

  @abstractmethod
  def mayVectorizeDim(self, dim):
    pass

  def mayFuse(self, positions):
    return len(positions) == 1
  
  @classmethod
  @abstractmethod
  def fromSpp(cls, spp, **kwargs):
    pass

  @abstractmethod
  def __contains__(self, entry):
    pass

  @abstractmethod
  def __eq__(self, other):
    pass

  @abstractmethod
  def isCompatible(self, spp):
    pass


class DenseMemoryLayout(MemoryLayout):
  ALIGNMENT_ARCH = None

  @classmethod
  def setAlignmentArch(cls, arch):
    cls.ALIGNMENT_ARCH = arch
  
  def __init__(self, shape, boundingBox=None, stride=None, alignStride=False):
    """
    Args:
      shape (Tuple[int, ...]): sizes of each tensor dimension
      boundingBox (Union[BoundingBox, None]): TODO
      stride (Union[Tuple[int, ...], None]): TODO
      alignStride (bool): TODO
    """
    super().__init__(shape)

    # init a bounding box of a dense memory layout
    if boundingBox:
      self._bbox = boundingBox
    else:

      # Create the bounding box of a dense memory layout from a tensor shape
      # if a bounding box was not provided by the user
      self._bbox = BoundingBox([Range(0, size) for size in self._shape])

    # align the first tensor dimension if vectorization is required
    self._range0 = None
    if alignStride:
      self._alignBB()


    # init stride of each tensor dimension
    if stride:
      self._stride = stride
    else:
      self._computeStride()


  def _computeStride(self):
    """Compute strides of each dimension of a tensor i.e. distances of each dimension
    the first tensor element

    Examples:
      >>> from yateto.memory import DenseMemoryLayout
      >>> layout = DenseMemoryLayout(shape=(3,4,3))
      >>> layout._computeStride()
      >>> layout.stride()
      (1, 3, 12)
    """
    stride = [1]
    for i in range(len(self._bbox) - 1):
      stride.append(stride[i] * self._bbox[i].size())
    self._stride = tuple(stride)


  def _alignBB(self):
    """Aligns the first dimension of a dense memory layout to enable vectorization
    """
    if self.ALIGNMENT_ARCH is not None:

      # extract a range of the first dimension
      self._range0 = self._bbox[0]

      # align the first dimention according to the given compute architecture
      new_leading_range = Range(self.ALIGNMENT_ARCH.alignedLower(self._range0.start),
                                self.ALIGNMENT_ARCH.alignedUpper(self._range0.stop))

      # substitude the old first dimnesion with a new (aligned) one
      self._bbox = BoundingBox([new_leading_range] + self._bbox[1:])
    else:
      warnings.warn('Set architecture with DenseMemoryLayout.setAlignmentArch(arch) '
                    'if you want to use the align stride feature.', UserWarning)


  def alignedStride(self):
    if self.ALIGNMENT_ARCH is None:
      return False

    offsetOk = self.ALIGNMENT_ARCH.checkAlignment(self._bbox[0].start)
    ldOk = self._stride[0] == 1 and (len(self._stride) == 1 or self.ALIGNMENT_ARCH.checkAlignment(self._stride[1]))
    return offsetOk and ldOk


  def mayVectorizeDim(self, dim):
    if self.ALIGNMENT_ARCH is None:
      return False
    return self.ALIGNMENT_ARCH.checkAlignment(self._bbox[dim].size())


  @classmethod
  def fromSpp(cls, spp, alignStride=False):
    bbox = BoundingBox.fromSpp(spp)
    return cls(spp.shape, bbox, alignStride=alignStride)


  def __contains__(self, entry):
    return entry in self._bbox


  def permuted(self, permutation):
    newShape = tuple([self._shape[p] for p in permutation])
    
    originalBB = BoundingBox([self._range0] + self._bbox[1:]) if self._range0 else self._bbox
    newBB = BoundingBox([copy.copy(originalBB[p]) for p in permutation])
    return DenseMemoryLayout(newShape, newBB, alignStride=self._range0 is not None)


  def address(self, element_index_set):
    """Compute a linearized element index, also called an element address, given an element
    as a set of indices within a tensor memory layout

    Args:
      element_index_set (Tuple[int, ...]): an index set of a particular tensor element

    Returns:
      int: a linearized element index

    Examples:
      >>> tensor_shape = (3,2,4)
      >>> memory_layout = DenseMemoryLayout(shape=tensor_shape)
      >>> element_indices = (2,2,1)
      >>> memory_layout.address(element_indices)
      14
    """
    assert element_index_set in self._bbox

    element_address = 0
    for counter, index in enumerate(element_index_set):
      element_address += (index - self._bbox[counter].start) * self._stride[counter]

    return element_address


  def subtensorOffset(self, topLeftEntry):
    return self.address(topLeftEntry)


  def notWrittenAddresses(self, writeBB):
    """Computes addresses of elements which are outside of a given memory layout region (writeBB)

    Args:
      writeBB (BoundingBox): a sub-tensor space which the user wants to write data in

    Returns:
      List[]: a list of linearized element tensor indices which are outside of writable tensor
              space

    Examples:
      >>> from yateto.ast.indices import BoundingBox
      >>> from yateto.ast.indices import Range
      >>> range_1 = Range(start=1, stop=3)
      >>> range_2 = Range(start=2, stop=4)
      >>> box = BoundingBox([range_1, range_2])
      >>> from yateto.memory import DenseMemoryLayout
      >>> tensor_shape = (5, 6)
      >>> memory_layout = DenseMemoryLayout(shape=tensor_shape)
      >>> memory_layout.notWrittenAddresses(box)
      [13, 0, 3, 29, 7, 21, 6, 25, 14, 27, 15, 1, 22, 4, 28, 5, 18, 24, 8, 26, 2, 20, 19, 23, 9, 10]
    """

    # return an empty list if a passed bounding box matches to a bounding box
    # of the current instance of a dense memory layout instance
    if writeBB == self._bbox:
      return []


    # ensure that all ranges of a passed bounding box are included in  (inside of)
    # ranges of the current memory layout
    assert writeBB in self._bbox

    # get all ranges of the current memory layout
    layout_ranges = [range(box_range.start, box_range.stop) for box_range in self._bbox]

    # get all ranges of the current memory layout
    write_box_ranges = [range(box_range.start, box_range.stop) for box_range in writeBB]

    # compute indices of a memory layout which is outside of a given (passed)
    # bounding box
    outside_indices = set(itertools.product(*layout_ranges)) \
                      - set(itertools.product(*write_box_ranges))

    return [self.address(index_set) for index_set in outside_indices]


  def stride(self):
    return self._stride


  def stridei(self, dim):
    return self._stride[dim]


  def bbox(self):
    return self._bbox


  def bboxi(self, dim):
    return self._bbox[dim]


  def requiredReals(self):
    """
    # TODO: check this line of the code
    size = self._bbox[-1].size() * self._stride[-1]
    return size
    """
    if len(self._bbox) == 0:
      return 1

    size = self._bbox[-1].size() * self._stride[-1]
    return size


  def addressString(self, indices, specific_names=None):
    """Generate a string with a linearized address within a tensor for a source code

    NOTE: it is widely used inside of nested for-loops. Renamed indices are iterable variables of
    nested for-loops

    Args:
      indices (Indices): a description of tensor indices
      specific_names (Optional[str]): TODO

    Returns:
      str: a linearized address of a tensor within a source code

    Examples:
      >>> from yateto.memory import DenseMemoryLayout
      >>> tensor_shape = (5, 6)
      >>> memory_layout = DenseMemoryLayout(shape=tensor_shape)
      >>> from yateto.ast.indices import Indices
      >>> indices = Indices(indexNames='ab', shape=(5, 6))
      >>> memory_layout.addressString(indices)
      '1*a + 5*b'

      >>> from yateto.memory import DenseMemoryLayout
      >>> tensor_shape = (5, 6)
      >>> memory_layout = DenseMemoryLayout(shape=tensor_shape)
      >>> from yateto.ast.indices import Indices
      >>> indices = Indices(indexNames='ab', shape=(5, 6))
      >>> memory_layout.addressString(indices, specific_names='b')
      '5*b'

    """

    # extract names of renamed indices of an instance of Indices
    if specific_names is None:
      names = set(indices)
    else:
      names = specific_names

    positions = indices.positions(names)

    address_parts = list()
    for index_position in positions:

      if self._bbox[index_position].start != 0:
        address_parts.append('{}*({}-{})'.format(self._stride[index_position],
                                                 indices[index_position],
                                                 self._bbox[index_position].start))
      else:
        address_parts.append('{}*{}'.format(self._stride[index_position],
                                            indices[index_position]))

    address = ' + '.join(address_parts)
    return address


  def isAlignedAddressString(self, indices, I = None):
    if I is None:
      I = set(indices)
    positions = indices.positions(I)
    for p in positions:
      if self.ALIGNMENT_ARCH.checkAlignment(self._stride[p]) == False:
        return False
    return True


  def mayFuse(self, positions):
    return all( [self._stride[j] == self._shape[i]*self._stride[i] for i,j in zip(positions[:-1], positions[1:])] )


  def _subShape(self, positions):
    sub = 1
    for p in positions:
      sub *= self._shape[p]
    return sub
  
  def _subRange(self, positions):
    start = 0
    stop = 0
    s = 1
    for p in positions:
      start += s * self._bbox[p].start
      stop += s * (self._bbox[p].stop-1)
      s *= self._shape[p]
    return Range(start, stop+1)
    
  def _firstStride(self, positions):
    return self._stride[ positions[0] ]

  def vec(self, indices, I):
    positionsI = indices.positions(I)
    assert self.mayFuse( indices.positions(I) )

    shape = (self._subShape(positionsI),)
    bbox = BoundingBox([self._subRange(positionsI)])
    stride = (self._firstStride(positionsI),)

    return DenseMemoryLayout(shape, bbox, stride)

  def withDummyDimension(self):
    shape = self._shape + (1,)
    bbox = BoundingBox(list(self._bbox) + [Range(0,1)])
    stride = self._stride + (self._bbox[-1].size() * self._stride[-1],)
    return DenseMemoryLayout(shape, bbox, stride)

  def unfold(self, indices, I, J):
    positionsI = indices.positions(I)
    positionsJ = indices.positions(J)
    assert self.mayFuse( indices.positions(I) ) and self.mayFuse( indices.positions(J) )

    if positionsI[0] > positionsJ[0]:
      positionsJ, positionsI = positionsI, positionsJ

    shape = (self._subShape(positionsI), self._subShape(positionsJ))
    bbox = BoundingBox([self._subRange(positionsI), self._subRange(positionsJ)])
    stride = (self._firstStride(positionsI), self._firstStride(positionsJ))

    return DenseMemoryLayout(shape, bbox, stride)
  
  def defuse(self, fusedRange, indices, I):
    positions = indices.positions(I)
    s = self._subShape(positions)
    ranges = dict()
    start = fusedRange.start
    stop = fusedRange.stop-1
    for p in reversed(positions):
      s //= self._shape[p]
      b = start // s
      B = stop // s
      ranges[ indices[p] ] = Range(b, B+1)
      start -= b*s
      stop -= B*s
    return ranges

  def isCompatible(self, spp):
    return BoundingBox.fromSpp(spp) in self.bbox()

  def __eq__(self, other):
    return self._stride == other._stride and self._bbox == other._bbox and self._stride == other._stride

  def __str__(self):
    return '{}(shape: {}, bounding box: {}, stride: {})'.format(type(self).__name__, self._shape, self._bbox, self._stride)


class CSCMemoryLayout(MemoryLayout):
  def __init__(self, spp):
    super().__init__(spp.shape)
    
    if len(self._shape) != 2:
      raise ValueError('CSCMemoryLayout may only be used for matrices.')
    
    self._bbox = BoundingBox.fromSpp(spp)
    
    nonzeros = spp.nonzero()
    nonzeros = sorted(zip(nonzeros[0], nonzeros[1]), key=lambda x: (x[1], x[0]))
    
    self._rowIndex = np.ndarray(shape=(len(nonzeros),), dtype=int)
    self._colPtr = np.ndarray(shape=(self._shape[1]+1,), dtype=int)
    
    lastCol = 0
    self._colPtr[0] = 0
    for i,entry in enumerate(nonzeros):
      self._rowIndex[i] = entry[0]
      if entry[1] != lastCol:
        for j in range(lastCol+1, entry[1]+1):
          self._colPtr[ j ] = i
        lastCol = entry[1]
    for j in range(lastCol+1, self._shape[1]+1):
      self._colPtr[j] = len(nonzeros)

  def requiredReals(self):
    return len(self._rowIndex)

  def bboxi(self, dim):
    return self._bbox[dim]
  
  def rowIndex(self):
    return self._rowIndex
  
  def colPointer(self):
    return self._colPtr
  
  def address(self, entry):
    assert entry in self._bbox

    start = self._colPtr[ entry[1] ]
    stop = self._colPtr[ entry[1]+1 ]
    subRowInd = self._rowIndex[start:stop]
 
    find = np.where(subRowInd == entry[0])[0]
    assert len(find) == 1

    return start + find[0]
  
  def subtensorOffset(self, topLeftEntry):
    assert topLeftEntry in self._bbox
    assert topLeftEntry[0] <= self._bbox[0].start
    return self._colPtr[ topLeftEntry[1] ]

  def entries(self, rowRange, colRange):
    assert self._bbox[0].start >= rowRange.start
    e = list()
    for col in colRange:
      e.extend([(self._rowIndex[i]-rowRange.start, col-colRange.start) for i in range(self._colPtr[col], self._colPtr[col+1])])
    return e

  def alignedStride(self):
    return False

  def mayVectorizeDim(self, dim):
    return False

  @classmethod
  def fromSpp(cls, spp, **kwargs):
    return CSCMemoryLayout(spp)

  def __contains__(self, entry):
    return entry in self._bbox

  def isCompatible(self, spp):
    return self.fromSpp(spp) == self

  def __eq__(self, other):
    return self._bbox == other._bbox and np.array_equal(self._rowIndex, other._rowIndex) and np.array_equal(self._colPtr, other._colPtr)
