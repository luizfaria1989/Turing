# conf.py
import os
import sys
sys.path.insert(0, os.path.abspath('../..')) # Aponta para a raiz do projeto

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Turing'
copyright = '2025, Luiz Faria'
author = 'Luiz Faria'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# conf.py
extensions = [
    'sphinx.ext.autodoc',      # Puxa a documentação das docstrings
    'sphinx.ext.napoleon',     # Entende os estilos de docstring do Google e NumPy
    'sphinx.ext.viewcode',     # Adiciona links para o código fonte
    'sphinx.ext.todo',         # Permite usar notas de "to-do"
]

html_theme = 'furo'

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
