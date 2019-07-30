import copy
from .node import LoopOverGEMM
from .indices import LoGCost

def allSubstrings(s):
  L = len(s)
  return [s[i:j+1] for i in range(L) for j in range(i,L)]


def splitByDistance(addresses):
  """Splits non-contiguous addresses into chunks of contiguous ones

  Args:
    addresses (List[int]): non-contiguous addresses of a memory layout

  Returns:
    List[List[int]]: contiguous chunks of addresses
  """
  num_addresses = len(addresses)
  splits = [counter + 1 for previous_address, next_address, counter
            in zip(addresses[:-1], addresses[1:], range(num_addresses))
            if next_address - previous_address != 1]

  return [addresses[i:j] for i, j in zip([0] + splits, splits + [num_addresses])]


def fusedVariants(memLayout, I, P, M, prune = False):
  D = list()
  indices = sorted([P[p] for p in I])
  groups = splitByDistance(indices)
  groupStrings = [''.join([M[p] for p in sorted(g)]) for g in groups]
  D = set([s for g in groupStrings for s in allSubstrings(g)])
  if prune:
    D = set([d for d in D if d[0] == M[0]])
  D = set([d for d in D if memLayout.mayFuse(sorted([P[i] for i in d]))])  
  return D

def LoG(contraction, Aperm = None, Bperm = None, Cperm = None):
  L = contraction.leftTerm()
  R = contraction.rightTerm()
  I = contraction
  
  if Aperm is not None:
    L = copy.copy(L)
    L.setIndexPermutation(Aperm, permuteEqspp=False)
  if Bperm is not None:
    R = copy.copy(R)
    R.setIndexPermutation(Bperm, permuteEqspp=False)
  if Cperm is not None:
    I = copy.copy(contraction)
    I.setIndexPermutation(Cperm, permuteEqspp=False)

  A = L.indices.tostring()
  B = R.indices.tostring()
  C = I.indices.tostring()

  candidates = list()
  requiredIndices = set([A[0], B[0], C[0]])
  if C[0] in set(B):
    B, A = A, B
    R, L = L, R
  Icommon = set(A) & set(B) & set(C)
  Im = (set(A) & set(C)) - Icommon
  In = (set(B) & set(C)) - Icommon
  Ik = (set(A) & set(B)) - Icommon
  
  PA = {idx: pos for pos, idx in enumerate(A)}
  PB = {idx: pos for pos, idx in enumerate(B)}
  PC = {idx: pos for pos, idx in enumerate(C)}

  CM = fusedVariants(I.memoryLayout(), Im, PC, C, True)
  CN = fusedVariants(I.memoryLayout(), In, PC, C)
  AM = fusedVariants(L.memoryLayout(), Im, PA, A)
  AK = fusedVariants(L.memoryLayout(), Ik, PA, A)
  BK = fusedVariants(R.memoryLayout(), Ik, PB, B)
  BN = fusedVariants(R.memoryLayout(), In, PB, B)
  
  MC = CM & AM
  NC = CN & BN
  KC = AK & BK

  if NC == set():
    NC = ['']

  minCost = LoGCost()
  minLog = None
  for m in MC:
    for n in NC:
      for k in KC:
        log = LoopOverGEMM(I.indices, L, R, m, n, k)
        cost = log.cost()
        if cost < minCost:
          minCost = cost
          minLog = log
  if minLog:
    minLog.setMemoryLayout( I.memoryLayout() )
  return minLog
