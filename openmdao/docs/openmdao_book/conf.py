"""Sphinx configuration for OpenMDAO documentation."""

project = 'OpenMDAO'
author = 'The OpenMDAO Development Team'
copyright = 'OpenMDAO Development Team'
root_doc = 'main'

extensions = [
    'myst_nb',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx_sitemap',
    'numpydoc',
    'sphinxcontrib.bibtex',
    'sphinx_copybutton',
    'sphinx_design',
]

# MyST / myst-nb settings
myst_enable_extensions = [
    'amsmath',
    'colon_fence',
    'dollarmath',
    'linkify',
    'substitution',
]

# Do not re-execute notebooks; use the pre-executed outputs from papermill
nb_execution_mode = 'off'
nb_output_stderr = 'show'

# Theme
html_theme = 'om-theme'
html_theme_path = ['.']   # om-theme/ lives in the same directory as conf.py
html_static_path = ['_static']
html_theme_options = {
    'navigation_with_keys': False,
}

# Site settings
html_baseurl = 'https://openmdao.org/newdocs/versions/latest/'
html_logo = 'OpenMDAO_Logo.png'
html_extra_path = []

# GitHub buttons
html_context = {
    'github_url': 'https://github.com/OpenMDAO/OpenMDAO',
    'github_repo': 'OpenMDAO',
    'github_user': 'OpenMDAO',
    'github_version': 'master',
    'doc_path': 'openmdao/docs/openmdao_book',
}

# BibTeX
bibtex_bibfiles = ['references.bib']

# Sitemap
sitemap_filename = 'sitemap.xml'

# Autodoc / autosummary
autosummary_generate = True
numpydoc_class_members_toctree = False
suppress_warnings = ['autodoc', 'autodoc.import_object']
autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'inherited-members': False,
    'show-inheritance': True,
}

# Exclude patterns
exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
    '**/.ipynb_checkpoints',
    'template.ipynb',
]
