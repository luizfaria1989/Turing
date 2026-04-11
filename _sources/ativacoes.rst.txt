Funções de Ativação
=====================

As funções de ativação são componentes matemáticos cruciais em redes neurais artificiais. Elas são responsáveis por introduzir **não-linearidade** ao modelo, permitindo que a rede aprenda e mapeie relações complexas nos dados. Sem elas, não importa quantas camadas ocultas a rede possua, ela se comportaria apenas como uma gigantesca regressão linear.

Abaixo, documentamos as funções de ativação implementadas na biblioteca Turing.

Rectified Linear Unit (ReLU)
-----------------------------

.. doxygenclass:: ReLU
   :project: Turing
   :members:

Sigmoide Logística (Sigmoid)
------------------------------

.. doxygenclass:: Sigmoid
   :project: Turing
   :members:

Tangente Hiperbólica (Tanh)
----------------------------
.. doxygenclass:: Tanh
   :project: Turing
   :members:

Softmax
----------

.. doxygenclass:: Softmax
   :project: Turing
   :members: