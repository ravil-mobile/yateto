##
# @file
# This file is part of SeisSol.
#
# @author Carsten Uphoff (c.uphoff AT tum.de, http://www5.in.tum.de/wiki/index.php/Carsten_Uphoff,_M.Sc.)
#
# @section LICENSE
# Copyright (c) 2015-2018, SeisSol Group
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from this
#    software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# @section DESCRIPTION
#

from .memory import DenseMemoryLayout


class Architecture(object):
  def __init__(self, name, precision, alignment, enablePrefetch=False):
    """
    Args:
      name (str): an compute architecture name
      precision (str): a character which describes precision to be used during source code
                       generation i.e. 's' - single; 'd' - double
      alignment (int): a vector register size in bytes
      enablePrefetch (bool): a flag which tells yateto whether generate a compute architecture
                             is able to prefetch data from memory

    Raises:
      ValueError: if provided precision is neither 'd' nor 's' i.e. neither double or single,
                  respectively.
    """
    self.name = name
    self.precision = precision.upper()
    if self.precision == 'D':
      self.bytesPerReal = 8
      self.typename = 'double'
      self.epsilon = 2.22e-16


    elif self.precision == 'S':
      self.bytesPerReal = 4
      self.typename = 'float'
      self.epsilon = 1.19e-7
    else:
      raise ValueError('Unknown precision type ' + self.precision)


    self.alignment = alignment
    assert self.alignment % self.bytesPerReal == 0
    self.alignedReals = self.alignment // self.bytesPerReal
    self.enablePrefetch = enablePrefetch
    
    self.uintTypename = 'unsigned'
    self.ulongTypename = 'unsigned long'

    self._tmpStackLimit = 524288


  def setTmpStackLimit(self, tmpStackLimit):
    """Sets a stack size which is going to be assumed during the source code generation

    NOTE: the stack size will affect on whether temporary buffers will be allocated
          on heap or stack

    Args:
      tmpStackLimit (int): a size of stack
    """
    self._tmpStackLimit = tmpStackLimit


  def alignedLower(self, index):
    """Computes an appropriate lower value of a given index value suitable for vectorization
       for a particular compute architecture

    Args:
      index (int): an index value of a range

    Returns:
      int: adjusted index value

    Examples:
      >>> from yateto.arch import Architecture
      >>> arch = Architecture(name='hsw', precision='D', alignment=32)
      >>> arch.alignedLower(index=4)
      4
      >>> arch.alignedLower(index=3)
      0
    """
    return index - index % self.alignedReals


  def alignedUpper(self, index):
    """Computes an appropriate upper value of a given index value suitable for vectorization
       for a particular compute architecture

    Args:
      index (int): an index value of a range

    Returns:
      int: adjusted index value

    Examples:
      >>> from yateto.arch import Architecture
      >>> arch = Architecture(name='hsw', precision='D', alignment=32)
      >>> arch.alignedUpper(index=12)
      12
      >>> arch.alignedUpper(index=13)
      16

    """
    return index + (self.alignedReals - index % self.alignedReals) % self.alignedReals


  def alignedShape(self, shape):
    return (self.alignedUpper(shape[0]),) + shape[1:]


  def checkAlignment(self, offset):
    """Checks whether a given index (offset) is aligned with respect to a specific compute
    architecture

    NOTE: an index belong to a range which in its turn belongs to a tensor

    Args:
      offset (int): an index

    Returns:
      bool: True, if an index is aligned. Otherwise, False
    """
    return offset % self.alignedReals == 0


  def formatConstant(self, constant):
    return str(constant) + ('f' if self.precision == 'S' else '')


  def onHeap(self, numReals):
    """Checks whether an array has to be allocated on heap

    Args:
      numReals (int): a size of an array

    Returns:
      bool: True, if an array has to be allocated on heap
    """
    return (numReals * self.bytesPerReal) > self._tmpStackLimit


def getArchitectureIdentifiedBy(ident):
    """Creates a particular architecture object based on the input string.

    Args:
      ident (str): a string which describes the target precision (the first
                   character) and architecture type (using the rest of the characters)

    Returns:
      Architecture: a specific compute architecture
    """
    precision = ident[0].upper()
    name = ident[1:]

    # implementation of switch case construct in python
    # NOTE: Libxsmm currently supports prefetch only for KNL kernels
    arch = {
      'noarch': Architecture(name, precision, 16, False),
      'wsm':    Architecture(name, precision, 16, False),
      'snb':    Architecture(name, precision, 32, False),
      'hsw':    Architecture(name, precision, 32, False),
      'skx':    Architecture(name, precision, 64, True),
      'knc':    Architecture(name, precision, 64, False),
      'knl':    Architecture(name, precision, 64, True)
    }
    return arch[name]


def useArchitectureIdentifiedBy(ident):
    """ Creates an architecture object initialized according
    to the input string as well as initializes DenseMemoryLayout
    class with the created architecture object

    Args:
      ident (str): a string which describes the target precision (the first
                   character) and architecture type (using the rest of the characters)

    Returns:
      Architecture: a specific compute architecture
    """
    arch = getArchitectureIdentifiedBy(ident)
    DenseMemoryLayout.setAlignmentArch(arch)
    return arch
