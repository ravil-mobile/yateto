from .external_generator import FusedGemms


class Description(object):
  def __init__(self, gemm_list):
    self._gemm_list = gemm_list


def generator(arch, descr, target):
  if target == 'gpu':
    return FusedGemms(arch, descr)
  else:
    ValueError(f'expected a GPU target, given {target}')
