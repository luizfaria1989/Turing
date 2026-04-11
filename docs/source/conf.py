# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
# -- Informações do Projeto --------------------------------------------------
project = 'Turing'
copyright = '2026, Luiz Guilherme Faria'
author = 'Luiz Guilherme Faria'
release = '1.0'

extensions = [
    'breathe',
    'sphinx.ext.mathjax',
]

html_permalinks_icon = '<span>#</span>'
html_theme = 'sphinxawesome_theme'
html_static_path = ['_static']

html_theme_options = {
    "show_prev_next": True,
    "show_breadcrumbs": True,
}

breathe_projects = {
    "Turing": "../../xml"
}
breathe_default_project = "Turing"
