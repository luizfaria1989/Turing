Funções de Ativação
===================

As funções de ativação são componentes matemáticos cruciais em redes neurais artificiais. Elas são responsáveis por introduzir não-linearidade ao modelo, permitindo que a rede aprenda e mapeie relações complexas nos dados. Sem elas, não importa quantas camadas ocultas a rede possua, ela se comportaria apenas como uma gigantesca regressão linear.

Abaixo, documentamos as funções de ativação implementadas na biblioteca Turing.

Rectified Linear Unit (ReLU)
----------------------------

.. doxygenclass:: ReLU
   :project: Turing
   :members:

Exemplo de Aplicação
^^^^^^^^^^^^^^^^^^^^
Abaixo está um exemplo completo de como instanciar e treinar uma rede utilizando a ReLU:

.. literalinclude:: ../../models/DNNWithNAG.cpp
   :language: cpp
   :linenos:
   :lines: 49-54

Sigmoide Logística (Sigmoid)
----------------------------

.. doxygenclass:: Sigmoid
   :project: Turing
   :members:

Exemplo de Aplicação
^^^^^^^^^^^^^^^^^^^^
Abaixo está um exemplo completo de como instanciar e treinar uma rede utilizando a Sigmoide Logística:

.. literalinclude:: ../../models/DNNWithGD.cpp
   :language: cpp
   :linenos:
   :lines: 49-54

Tangente Hiperbólica (Tanh)
---------------------------

.. doxygenclass:: Tanh
   :project: Turing
   :members:

Exemplo de Aplicação
^^^^^^^^^^^^^^^^^^^^
Abaixo está um exemplo completo de como instanciar e treinar uma rede utilizando a Tangente Hiperbólica:

.. literalinclude:: ../../models/DNNWithGDMomentum.cpp
   :language: cpp
   :linenos:
   :lines: 49-54

Softmax
-------

.. doxygenclass:: Softmax
   :project: Turing
   :members:

Exemplo de Aplicação
^^^^^^^^^^^^^^^^^^^^
Abaixo está um exemplo completo de como instanciar e treinar uma rede utilizando a Softmax:

.. literalinclude:: ../../models/DNNWithGD.cpp
   :language: cpp
   :linenos:
   :lines: 49-58