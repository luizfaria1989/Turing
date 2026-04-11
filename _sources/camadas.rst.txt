Camadas (Layers)
===================

As camadas são os blocos de construção fundamentais de qualquer rede neural profunda.
Elas contêm os parâmetros treináveis do modelo (pesos e vieses) e são responsáveis por
aplicar transformações geométricas nos dados de entrada.

Na arquitetura da biblioteca Turing, o aprendizado de máquina é tratado como um pipeline
modular. O dado flui para frente através das camadas (Forward Pass) para gerar uma previsão,
e o erro flui de trás para frente (Backward Pass) ajustando as matrizes de cada camada.

Interface Base (Layer)
-------------------------

.. doxygenclass:: Layer
   :project: Turing
   :members:

Camada Densa (DenseLayer)
--------------------------

.. doxygenclass:: DenseLayer
   :project: Turing
   :members: