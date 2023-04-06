# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Causal Inspection'
copyright = '2023, Gradient Institute'
author = 'Gradient Institute'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.napoleon',  # use numpy docstring format
              # 'sphinx.ext.coverage',  # -b coverage on cl gives documentation coverage
              'sphinx.ext.extlinks',
              #   'sphinx.ext.linkcode',  # for linking to external repo.
              # for potential use with github,
              # see https://github.com/readthedocs/sphinx-autoapi/issues/202
              # or https://github.com/scikit-image/scikit-image/pull/1628
              'sphinx.ext.viewcode',  # embeds source code in generated html
              'sphinx.ext.intersphinx',  # link to external docs for dependencies
              ]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for sphinx.ext.autodoc

autoclass_content = 'class'  # 'both': concatenate, display *both* the class and __init__ docstrings
autodoc_typehints = "both" # make explicit that typehints are shown in the signature, rather than the description
# -- Options for sphinx.ext.extlinks

# use :issue:`123` to link to project issue on GitHub
extlinks = {'issue': ('https://github.com/gradientinstitute/cinspect/issues/%s',
                      'issue %s')}


# -- Options for sphinx.ext.intersphinx

intersphinx_mapping = {'python': ('https://docs.python.org/3', None),
                       'scikit-learn': ('https://scikit-learn.org/stable/', None),
                       'pandas': ('https://pandas.pydata.org/docs/', None),
                       'numpy': ('https://numpy.org/doc/stable/', None),
                       'matplotlib': ('https://matplotlib.org/stable/', None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
