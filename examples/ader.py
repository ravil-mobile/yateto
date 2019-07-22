from yateto.type import Tensor
from yateto.ast.node import IndexedTensor
from yateto.memory import DenseMemoryLayout

####################################################################################################
class OptionalDimTensor(Tensor):
  # dimSize = 1 is considered optional
  def __init__(self,
               name,
               optName,
               optSize,
               optPos,
               shape,
               spp=None,
               memoryLayoutClass=DenseMemoryLayout,
               alignStride=False):
    """

    Args:
      name (str): a tensor name
      optName (str): TODO
      optSize (int): number of simulations stacked together
      optPos (int): TODO
      shape (Tuple[int]): base tensor shape i.e. a shape for one simulation
      spp (): TODO
      memoryLayoutClass ():
      alignStride (bool): TODO
    """

    self._optName = optName
    self._optSize = optSize
    self._optPos = optPos
    shape = self.insertOptDim(shape, (self._optSize,))
    super().__init__(name, shape, spp, memoryLayoutClass, alignStride)


  def hasOptDim(self):
    return self._optSize > 1


  def insertOptDim(self, sliceable, item):
    if self.hasOptDim():
      return sliceable[0:self._optPos] + item + sliceable[self._optPos:]
    return sliceable


  def __getitem__(self, indexNames):
    indexNames = self.insertOptDim(indexNames, self._optName)
    return IndexedTensor(self, indexNames)


  def optName(self):
    return self._optName


  def optSize(self):
    return self._optSize


  def optPos(self):
    return self._optPos


####################################################################################################
from yateto.input import parseXMLMatrixFile
from abc import ABC, abstractmethod

class ADERDGBase(ABC):
  def __init__(self, order, multipleSimulations, matricesDir):
    self.order = order

    self.alignStride = lambda name: True

    if multipleSimulations > 1:
      self.alignStride = lambda name: name.startswith('fP')

    transpose = multipleSimulations > 1
    self.transpose = lambda name: transpose

    # define a helper function which allows to flip indices of a tensor
    # if "transpose" is True e.g.
    # >>> self.t('abc')
    # 'cba'
    self.t = (lambda x: x[::-1]) if transpose else (lambda x: x)

    # read matrices from a file and add it to a collection i.e. db - database
    path_to_file = '{}/matrices_{}.xml'.format(matricesDir, self.numberOf3DBasisFunctions())
    self.db = parseXMLMatrixFile(xml_file=path_to_file,
                                 transpose=self.transpose,
                                 align_stride=self.alignStride)



    path_to_file = '{}/plasticity_ip_matrices_{}.xml'.format(matricesDir, order)

    # matrices that have to be renamed i.e. a key is a name of matrix in the file, a value is
    # a target matrix(tensor) name.
    # NOTE if a list (value) contains more than one entry, yateto will generate tensors
    # with the same content and structure but with different names according to the names
    # specified in the list
    clonesQP = {'v': ['evalAtQP'], 'vInv': ['projectQP']}

    # read matrices from a file and augment the collection i.e. db - database
    self.db.update(parseXMLMatrixFile(xml_file=path_to_file,
                                      clones=clonesQP,
                                      transpose=self.transpose,
                                      align_stride=self.alignStride))


    qShape = (self.numberOf3DBasisFunctions(), self.numberOfQuantities())
    self.Q = OptionalDimTensor(name='Q',
                               optName='s',
                               optSize=multipleSimulations,
                               optPos=0,
                               shape=qShape,
                               alignStride=True)

    self.I = OptionalDimTensor(name='I',
                               optName='s',
                               optSize=multipleSimulations,
                               optPos=0,
                               shape=qShape,
                               alignStride=True)


    Ashape = (self.numberOfQuantities(), self.numberOfExtendedQuantities())
    self.AplusT = Tensor(name='AplusT', shape=Ashape)
    self.AminusT = Tensor(name='AminusT', shape=Ashape)


    Tshape = (self.numberOfExtendedQuantities(), self.numberOfExtendedQuantities())
    self.T = Tensor(name='T', shape=Tshape)


    QgodShape = (self.numberOfQuantities(), self.numberOfQuantities())
    self.Tinv = Tensor(name='Tinv', shape=QgodShape)
    self.QgodLocal = Tensor(name='QgodLocal', shape=QgodShape)
    self.QgodNeighbor = Tensor(name='QgodNeighbor', shape=QgodShape)

    self.oneSimToMultSim = Tensor(name='oneSimToMultSim',
                                  shape=(self.Q.optSize(),),
                                  spp={(i,): '1.0' for i in range(self.Q.optSize())})


  def numberOf2DBasisFunctions(self):
    return self.order * (self.order + 1) // 2


  def numberOf3DBasisFunctions(self):
    return self.order * (self.order + 1) * (self.order + 2) // 6


  def numberOf3DQuadraturePoints(self):
    return (self.order + 1)**3


  @abstractmethod
  def numberOfQuantities(self):
    pass


  @abstractmethod
  def numberOfExtendedQuantities(self):
    pass


  @abstractmethod
  def extendedQTensor(self):

    pass

  @abstractmethod
  def starMatrix(self, dim):
    pass

  def addInit(self, generator):
    fluxScale = Scalar('fluxScale')
    computeFluxSolverLocal = self.AplusT['ij'] <= fluxScale * self.Tinv['ki'] * self.QgodLocal['kq'] * self.db.star[0]['ql'] * self.T['jl']
    generator.add(name='computeFluxSolverLocal', ast=computeFluxSolverLocal)

    computeFluxSolverNeighbor = self.AminusT['ij'] <= fluxScale * self.Tinv['ki'] * self.QgodNeighbor['kq'] * self.db.star[0]['ql'] * self.T['jl']
    generator.add(name='computeFluxSolverNeighbor', ast=computeFluxSolverNeighbor)

    QFortran = Tensor(name='QFortran',
                      shape=(self.numberOf3DBasisFunctions(), self.numberOfQuantities()))

    multSimToFirstSim = Tensor(name='multSimToFirstSim',
                               shape=(self.Q.optSize(),),
                               spp={(0,): '1.0'})

    if self.Q.hasOptDim():
      copyQToQFortran = QFortran['kp'] <= self.Q['kp'] * multSimToFirstSim['s']
    else:
      copyQToQFortran = QFortran['kp'] <= self.Q['kp']

    generator.add(name='copyQToQFortran', ast=copyQToQFortran)


  @abstractmethod
  def addLocal(self, generator):
    pass

  @abstractmethod
  def addNeighbor(self, generator):
    pass

  @abstractmethod
  def addTime(self, generator):
    pass



####################################################################################################
from yateto import Scalar, simpleParameterSpace
from yateto.input import parseXMLMatrixFile, memoryLayoutFromFile
from yateto.ast.node import Add
from yateto.ast.transformer import DeduceIndices, EquivalentSparsityPattern


class ADERDG(ADERDGBase):
  def __init__(self, order, multipleSimulations, matricesDir, memLayout):
    super().__init__(order, multipleSimulations, matricesDir)
    clones = {
      'star': ['star(0)', 'star(1)', 'star(2)'],
    }

    path_to_file = '{}/star.xml'.format(matricesDir)
    self.db.update(parseXMLMatrixFile(xml_file=path_to_file, clones=clones))

    # NOTE: this line has no effect on dense matrix layout
    memoryLayoutFromFile(xml_file=memLayout, db=self.db, clones=clones)

  def numberOfQuantities(self):
    return 9

  def numberOfExtendedQuantities(self):
    return self.numberOfQuantities()

  def extendedQTensor(self):
    return self.Q

  def starMatrix(self, dim):
    return self.db.star[dim]


  def addInit(self, generator):
    super().addInit(generator)

    iniShape = (self.numberOf3DQuadraturePoints(), self.numberOfQuantities())

    iniCond = OptionalDimTensor(name='iniCond',
                                optName=self.Q.optName(),
                                optSize=self.Q.optSize(),
                                optPos=self.Q.optPos(),
                                shape=iniShape,
                                alignStride=True)

    dofsQP = OptionalDimTensor(name='dofsQP',
                               optName=self.Q.optName(),
                               optSize=self.Q.optSize(),
                               optPos=self.Q.optPos(),
                               shape=iniShape,
                               alignStride=True)


    generator.add(name='projectIniCond',
                  ast=self.Q['kp'] <= self.db.projectQP[self.t('kl')] * iniCond['lp'])

    generator.add(name='evalAtQP',
                  ast=dofsQP['kp'] <= self.db.evalAtQP[self.t('kl')] * self.Q['lp'])


  def addLocal(self, generator):
    volumeSum = self.Q['kp']

    for i in range(3):
      volumeSum += self.db.kDivM[i][self.t('kl')] * self.I['lq'] * self.db.star[i]['qp']

    volume = (self.Q['kp'] <= volumeSum)
    generator.add(name='volume', ast=volume)

    localFlux = lambda i: self.Q['kp'] <= self.Q['kp'] + self.db.rDivM[i][self.t('km')] * self.db.fMrT[i][self.t('ml')] * self.I['lq'] * self.AplusT['qp']
    localFluxPrefetch = lambda i: self.I if i == 0 else (self.Q if i == 1 else None)

    generator.addFamily(name='localFlux',
                        parameterSpace=simpleParameterSpace(4),
                        astGenerator=localFlux,
                        prefetchGenerator=localFluxPrefetch)


  def addNeighbor(self, generator):
    neighbourFlux = lambda h,j,i: self.Q['kp'] <= self.Q['kp'] + self.db.rDivM[i][self.t('km')] * self.db.fP[h][self.t('mn')] * self.db.rT[j][self.t('nl')] * self.I['lq'] * self.AminusT['qp']
    neighbourFluxPrefetch = lambda h,j,i: self.I

    generator.addFamily(name='neighboringFlux',
                        parameterSpace=simpleParameterSpace(3,4,4),
                        astGenerator=neighbourFlux,
                        prefetchGenerator=neighbourFluxPrefetch)


  def addTime(self, generator):
    qShape = (self.numberOf3DBasisFunctions(), self.numberOfQuantities())
    dQ0 = OptionalDimTensor(name='dQ(0)',
                            optName=self.Q.optName(),
                            optSize=self.Q.optSize(),
                            optPos=self.Q.optPos(),
                            shape=qShape,
                            alignStride=True)

    power = Scalar('power')
    derivatives = [dQ0]

    generator.add(name='derivativeTaylorExpansion(0)',
                  ast=self.I['kp'] <= power * dQ0['kp'])

    for i in range(1, self.order):
      derivativeSum = Add()


      for j in range(3):
        derivativeSum += self.db.kDivMT[j][self.t('kl')] * derivatives[-1]['lq'] * self.db.star[j]['qp']


      derivativeSum = DeduceIndices(targetIndices=self.Q['kp'].indices).visit(derivativeSum)
      derivativeSum = EquivalentSparsityPattern().visit(derivativeSum)

      dQ = OptionalDimTensor(name='dQ({})'.format(i),
                             optName=self.Q.optName(),
                             optSize=self.Q.optSize(),
                             optPos=self.Q.optPos(),
                             shape=qShape,
                             spp=derivativeSum.eqspp(),
                             alignStride=True)


      generator.add(name='derivative({})'.format(i),
                    ast=dQ['kp'] <= derivativeSum)

      generator.add(name='derivativeTaylorExpansion({})'.format(i),
                    ast=self.I['kp'] <= self.I['kp'] + power * dQ['kp'])

      derivatives.append(dQ)



####################################################################################################
from yateto.gemm_configuration import GeneratorCollection, SeissolCudaBlas, MKL

def gemm_cfg(arch, variant=None):

  if variant == 'cuda':
    return GeneratorCollection([SeissolCudaBlas(arch)])
  else:
    return GeneratorCollection([MKL(arch)])


def add(generator):
  ader_dg = ADERDG(order=2,
                   multipleSimulations=1,
                   matricesDir="../../../generated_code/matrices",
                   memLayout="../../../auto_tuning/config/dense.xml")

  # Equation-specific kernels
  #ader_dg.addInit(generator)
  #ader_dg.addLocal(generator)
  #ader_dg.addNeighbor(generator)
  ader_dg.addTime(generator)