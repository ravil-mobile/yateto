class FusedGemms:
  def __init__(self, arch, descr):
    self._arch = arch
    self._descr = descr

  def generate(self, cpp, routineCache, cfg):
    print('calling gemmboost')
    return 0
