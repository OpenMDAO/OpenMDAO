# -*- coding: utf-8 -*-
# This file is execfile()d with the current directory set to its
# containing dir.
import sys, os
from mock import Mock
MOCK_MODULES = ['h5py', 'petsc4py', 'mpi4py', 'pyoptsparse']
sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)

import openmdao

def generate_docs(type):
    index_top_dev = """.. _source_documentation_dev:

=======================================
OpenMDAO Developer Source Documentation
=======================================

.. toctree::
   :titlesonly:
   :maxdepth: 1

"""
    index_top_usr = """.. _source_documentation_usr:

==================================
OpenMDAO User Source Documentation
==================================

.. toctree::
   :titlesonly:
   :maxdepth: 1

"""
    package_top = """
.. toctree::
    :maxdepth: 1

"""

    if(type == "dev"):
        ref_sheet_bottom = """
   :members:
   :undoc-members:
   :private-members:
   :show-inheritance:
   :inherited-members:
   :noindex:

.. toctree::
   :maxdepth: 1
"""
    else:
        ref_sheet_bottom = """
   :members:
   :noindex:

.. toctree::
   :maxdepth: 2
"""

    # need to set up the srcdocs directory structure, relative to docs.
    dir = os.path.dirname(__file__)
    if os.path.isdir(os.path.join(dir, "srcdocs", type)):
        import shutil
        shutil.rmtree(os.path.join(dir, "srcdocs", type))

    if not os.path.isdir(os.path.join(dir, "srcdocs", type)):
        os.mkdir(os.path.join(dir, "srcdocs", type))
    if not os.path.isdir(os.path.join(dir, "srcdocs", type, "packages")):
        os.mkdir(os.path.join(dir, "srcdocs", type, "packages"))

    # look for directories in the openmdao level, one up from docs
    # those directories will be the openmdao packages
    # auto-generate the top-level index.rst file for srcdocs, based on
    # openmdao packages:
    IGNORE_LIST = ['docs', 'tests', 'devtools', '__pycache__']
    # to improve the order that the user sees in the source docs, put
    # the important packages in this list explicitly. Any new ones that
    # get added will show up at the end.
    packages = ['assemblers','core', 'drivers', 'jacobians', 'solvers',
                'proc_allocators', 'vectors']
    # Everything in dir that isn't discarded is appended as a source package.
    for listing in os.listdir(os.path.join(dir, "..")):
        if os.path.isdir(os.path.join("..", listing)):
            if listing not in IGNORE_LIST and listing not in packages:
                packages.append(listing)

    # begin writing the 'srcdocs/index.rst' file at mid  level.
    index_filename = os.path.join(dir, "srcdocs", type, "index.rst")
    index = open(index_filename, "w")
    if (type == "dev"):
        index.write(index_top_dev)
    else:
        index.write(index_top_usr)

    # auto-generate package header files (e.g. 'openmdao.core.rst')
    for package in packages:
        # a package is e.g. openmdao.core, that contains source files
        # a sub_package, is a src file, e.g. openmdao.core.component
        sub_packages = []
        package_filename = os.path.join(dir, "srcdocs", type, "packages",
                                        "openmdao." + package + ".rst")
        package_name = "openmdao." + package

        # the sub_listing is going into each package dir and listing what's in it
        for sub_listing in sorted(os.listdir(os.path.join("..", package))):
            # don't want to catalog files twice, nor use init files nor test dir
            if (os.path.isdir(sub_listing) and sub_listing != "tests") or \
               (sub_listing.endswith(".py") and not sub_listing.startswith('_')):
                # just want the name of e.g. dataxfer not dataxfer.py
                sub_packages.append(sub_listing.rsplit('.')[0])

        if len(sub_packages) > 0:
            # continue to write in the top-level index file.
            # only document non-empty packages -- to avoid errors
            # (e.g. at time of writing, doegenerators, drivers, are empty dirs)

            #specifically don't use os.path.join here.  Even windows wants the
            #stuff in the file to have fwd slashes.
            index.write("   packages/openmdao." + package + "\n")

            # make subpkg directory (e.g. srcdocs/packages/core) for ref sheets
            package_dirname = os.path.join(dir, "srcdocs", type, "packages", package)
            os.mkdir(package_dirname)

            # create/write a package index file: (e.g. "srcdocs/packages/openmdao.core.rst")
            package_file = open(package_filename, "w")
            package_file.write(package_name + "\n")
            package_file.write("-" * len(package_name) + "\n")
            package_file.write(package_top)

            for sub_package in sub_packages:
                SKIP_SUBPACKAGES = []
                # this line writes subpackage name e.g. "core/component.py"
                # into the corresponding package index file (e.g. "openmdao.core.rst")
                if sub_package not in SKIP_SUBPACKAGES:
                    #specifically don't use os.path.join here.  Even windows wants the
                    #stuff in the file to have fwd slashes.
                    package_file.write("    " + package + "/" + sub_package + "\n")

                    # creates and writes out one reference sheet (e.g. core/component.rst)
                    ref_sheet_filename = os.path.join(package_dirname, sub_package + ".rst")
                    ref_sheet = open(ref_sheet_filename, "w")
                    # get the meat of the ref sheet code done
                    filename = sub_package + ".py"
                    ref_sheet.write(".. index:: " + type + "_" + filename + "\n\n")
                    ref_sheet.write(".. _" + type + "_" + package_name + "." + filename + ":\n\n")
                    ref_sheet.write(filename + "\n")
                    ref_sheet.write("+" * len(filename) + "\n\n")
                    ref_sheet.write(".. automodule:: " + package_name + "." + sub_package)

                    # finish and close each reference sheet.
                    ref_sheet.write(ref_sheet_bottom)
                    ref_sheet.close()


            # finish and close each package file
            package_file.close()

    # finish and close top-level index file
    index.close()

#generate two versions of the docs, one with private members, one without.
generate_docs("dev")
generate_docs("usr")

#------------------------begin monkeypatch-----------------------
#monkeypatch to make our docs say "Args" instead of "Parameters"
from numpydoc.docscrape_sphinx import SphinxDocString
from numpydoc.docscrape import NumpyDocString, Reader
import textwrap

def _parse(self):
        self._doc.reset()
        self._parse_summary()

        sections = list(self._read_sections())
        section_names = set([section for section, content in sections])

        has_returns = 'Returns' in section_names
        has_yields = 'Yields' in section_names
        # We could do more tests, but we are not. Arbitrarily.
        if has_returns and has_yields:
            msg = 'Docstring contains both a Returns and Yields section.'
            raise ValueError(msg)

        for (section, content) in sections:
            if not section.startswith('..'):
                section = (s.capitalize() for s in section.split(' '))
                section = ' '.join(section)
                if self.get(section):
                    msg = ("The section %s appears twice in the docstring." %
                           section)
                    raise ValueError(msg)

            if section in ('Args', 'Options', 'Params', 'Returns', 'Yields', 'Raises',
                           'Warns', 'Other Args', 'Attributes',
                           'Methods'):
                self[section] = self._parse_param_list(content)
            elif section.startswith('.. index::'):
                self['index'] = self._parse_index(section, content)
            elif section == 'See Also':
                self['See Also'] = self._parse_see_also(content)
            else:
                self[section] = content

def __str__(self, indent=0, func_role="obj"):
        out = []
        out += self._str_signature()
        out += self._str_index() + ['']
        out += self._str_summary()
        out += self._str_extended_summary()
        out += self._str_param_list('Args')
        out += self._str_options('Options')
        out += self._str_options('Params')
        out += self._str_returns()
        for param_list in ('Other Args', 'Raises', 'Warns'):
            out += self._str_param_list(param_list)
        out += self._str_warnings()
        out += self._str_see_also(func_role)
        out += self._str_section('Notes')
        out += self._str_references()
        out += self._str_examples()
        for param_list in ('Attributes', 'Methods'):
            out += self._str_member_list(param_list)
        out = self._str_indent(out,indent)
        return '\n'.join(out)

def __init__(self, docstring, config={}):

        docstring = textwrap.dedent(docstring).split('\n')

        self._doc = Reader(docstring)
        self._parsed_data = {
            'Signature': '',
            'Summary': [''],
            'Extended Summary': [],
            'Args': [],
            'Options': [],
            'Returns': [],
            'Raises': [],
            'Warns': [],
            'Other Args': [],
            'Attributes': [],
            'Params': [],
            'Methods': [],
            'See Also': [],
            'Notes': [],
            'Warnings': [],
            'References': '',
            'Examples': '',
            'index': {}
            }

        try:
            self._parse()
        except ParseError as e:
            e.docstring = orig_docstring
            raise

def _str_options(self, name):
        out = []
        if self[name]:
            out += self._str_field_list(name)
            out += ['']
            for param, param_type, desc in self[name]:
                if param_type:
                    out += self._str_indent(['**%s** : %s' % (param.strip(),
                                                              param_type)])
                else:
                    out += self._str_indent(['**%s**' % param.strip()])
                if desc:
                    out += ['']
                    out += self._str_indent(desc, 8)
                out += ['']
        return out

#Do the actual patch switchover to these local versions
NumpyDocString.__init__ = __init__
SphinxDocString._str_options = _str_options
SphinxDocString._parse = _parse
SphinxDocString.__str__ = __str__
#--------------end monkeypatch---------------------



# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('.'))
absp = os.path.join('..', 'srcdocs')
sys.path.insert(0, os.path.abspath(absp))

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'numpydoc'
]

numpydoc_show_class_members = False

#autodoc_default_flags = ['members']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = u'OpenMDAO'
copyright = u'2016, openmdao.org'
author = u'openmdao.org'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
#version = openmdao.__version__
version = "2.0.0"
# The full version, including alpha/beta/rc tags.
#release = openmdao.__version__ + ' Alpha'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'theme'
#html_theme = 'sphinxdoc'

# Add any paths that contain custom themes here, relative to this directory.
html_theme_path = ['.']

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = '_static/OpenMDAO_Logo.png'

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = '_static/OpenMDAO_Favicon.ico'

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
html_last_updated_fmt = '%b %d, %Y'

# Output file base name for HTML help builder.
htmlhelp_basename = 'OpenMDAOdoc'

# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'openmdao', u'OpenMDAO Documentation',
     [author], 1)
]
