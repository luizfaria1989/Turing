Funções de Perda (Loss)
========================

As funções de perda (ou funções de custo) atuam como o "juiz" durante o treinamento
da rede neural. Elas quantificam matematicamente a discrepância entre as previsões
feitas pelo modelo e os rótulos reais (ground truth) dos dados.

O objetivo central de todo o processo de aprendizado de máquina é minimizar o valor
escalar retornado por esta função através do cálculo de seus gradientes (Backpropagation)
e do reajuste iterativo dos pesos.

Abaixo, detalhamos a arquitetura base e as implementações específicas da biblioteca Turing.

Interface Base (Loss)
-----------------------

.. doxygenclass:: Loss
   :project: Turing
   :members:

Entropia Cruzada Categórica (CCE)
-------------------------------------
.. doxygenclass:: CategoricalCrossEntropy
   :project: Turing
   :members:

Exemplo de Aplicação
^^^^^^^^^^^^^^^^^^^^
Abaixo está um exemplo completo de como instanciar e treinar uma rede utilizando a entropia cruzada categórica como função de perdar:

.. literalinclude:: ../../models/DNNWithNAG.cpp
   :language: cpp
   :linenos:
   :lines: 49-75