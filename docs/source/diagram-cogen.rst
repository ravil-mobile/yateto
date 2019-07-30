Code Generation Module
======================

.. rst-class:: center

General
-------


Code
^^^^

.. inheritance-diagram:: yateto.codegen.code
    :parts: 2


Common
^^^^^^

.. inheritance-diagram:: yateto.codegen.common
    :parts: 2


Factories
^^^^^^^^^

.. inheritance-diagram:: yateto.codegen.factory yateto.codegen.cuda_factory
    :parts: 2



Visitors
^^^^^^^^

.. inheritance-diagram:: yateto.codegen.visitor yateto.codegen.cuda_visitor
    :parts: 2


General Matrix Matrix Multiplication
------------------------------------


Gemm Factory
^^^^^^^^^^^^

.. inheritance-diagram:: yateto.codegen.gemm.factory
    :parts: 2


Gemm Generic
^^^^^^^^^^^^

.. inheritance-diagram:: yateto.codegen.gemm.generic
    :parts: 2


Gemm Generator
^^^^^^^^^^^^^^

.. inheritance-diagram:: yateto.codegen.gemm.gemmgen
    :parts: 2


Loop Over Gemm
--------------

LOG cuda_factory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. inheritance-diagram:: yateto.codegen.log.factory
    :parts: 2


LOG Generic
^^^^^^^^^^^

.. inheritance-diagram:: yateto.codegen.log.generic
    :parts: 2