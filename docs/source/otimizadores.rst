Otimizadores
===================

Os otimizadores são o "motor de convergência" de uma rede neural. Enquanto a função de perda
nos diz *o quão errado* o modelo está, o otimizador dita a estratégia exata de **como** os parâmetros
(pesos e vieses) devem ser ajustados para corrigir esse erro.

Através das derivadas calculadas no *Backpropagation*, o algoritmo de otimização define as regras
para descer a superfície de erro passo a passo, em busca do mínimo global da função objetivo.

.. note::
   O hiperparâmetro mais crítico em qualquer otimizador é a **Taxa de Aprendizado** ($\eta$).
   Um valor muito alto pode fazer o modelo divergir e "explodir", enquanto um valor muito
   baixo fará o treinamento demorar uma eternidade ou travar em mínimos locais.

Interface Base (Optimizer)
----------------------------

.. doxygenclass:: Optimizer
   :project: Turing
   :members:

Gradiente Descedente (GradientDescent)
----------------------------------------

.. doxygenclass:: GradientDescent
   :project: Turing
   :members:

Exemplo de Aplicação
^^^^^^^^^^^^^^^^^^^^
Abaixo está um exemplo completo de como instanciar e treinar uma rede utilizando o gradiente descendente como otimizador:

.. literalinclude:: ../../models/DNNWithGD.cpp
   :language: cpp
   :linenos:
   :lines: 49-75

Gradiente Descendente com Momento (GDMomentum)
-----------------------------------------------
.. doxygenclass:: GDMomentum
   :project: Turing
   :members:

Exemplo de Aplicação
^^^^^^^^^^^^^^^^^^^^
Abaixo está um exemplo completo de como instanciar e treinar uma rede utilizando o gradiente descendente com momento como otimizador:

.. literalinclude:: ../../models/DNNWithGDMomentum.cpp
   :language: cpp
   :linenos:
   :lines: 49-75

Gradiente Acelerado de Nesterov (NAG)
---------------------------------------

.. doxygenclass:: NAG
   :project: Turing
   :members:

Exemplo de Aplicação
^^^^^^^^^^^^^^^^^^^^
Abaixo está um exemplo completo de como instanciar e treinar uma rede utilizando o gradiente acelerado de Nesterov como otimizador:

.. literalinclude:: ../../models/DNNWithNAG.cpp
   :language: cpp
   :linenos:
   :lines: 49-75
