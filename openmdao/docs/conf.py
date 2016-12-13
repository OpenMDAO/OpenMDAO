# -*- coding: utf-8 -*-
# This file is execfile()d with the current directory set to its
# containing dir.
import sys
import os


from mock import Mock
MOCK_MODULES = ['h5py', 'petsc4py', 'mpi4py', 'pyoptsparse']
sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)

import openmdao


# this function is used to create the entire directory structure
# of our source docs, as well as writing out each individual rst file.
def generate_docs(doctype):
    index_top_dev = """:orphan:

.. _source_documentation_dev:

=======================================
OpenMDAO Developer Source Documentation
=======================================

.. toctree::
   :titlesonly:
   :maxdepth: 1

"""
    index_top_usr = """:orphan:

.. _source_documentation_usr:

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

    if(doctype == "usr"):
        ref_sheet_bottom = """
   :members:
   :undoc-members:
   :special-members: __init__, __contains__, __iter__, __setitem__, __getitem__
   :show-inheritance:
   :inherited-members:

.. toctree::
   :maxdepth: 1
"""
    elif(doctype == "dev"):
        ref_sheet_bottom = """
   :members:
   :show-inheritance:
   :private-members:
   :special-members: __init__, __contains__, __iter__, __setitem__, __getitem__

.. toctree::
   :maxdepth: 1
"""

    # need to set up the srcdocs directory structure, relative to docs.
    docs_dir = os.path.dirname(__file__)

    doc_dir = os.path.join(docs_dir, "srcdocs", doctype)
    if os.path.isdir(doc_dir):
        import shutil
        shutil.rmtree(doc_dir)

    if not os.path.isdir(doc_dir):
        os.mkdir(doc_dir)

    packages_dir = os.path.join(doc_dir, "packages")
    if not os.path.isdir(packages_dir):
        os.mkdir(packages_dir)

    # look for directories in the openmdao level, one up from docs
    # those directories will be the openmdao packages
    # auto-generate the top-level index.rst file for srcdocs, based on
    # openmdao packages:
    IGNORE_LIST = [
        'docs', 'tests', 'devtools', '__pycache__', 'code_review', 'test_suite'
    ]

    # to improve the order that the user sees in the source docs, put
    # the important packages in this list explicitly. Any new ones that
    # get added will show up at the end.
    packages = [
        'assemblers', 'core', 'components', 'drivers', 'jacobians',
        'matrices', 'solvers', 'proc_allocators', 'utils', 'vectors'
    ]

    # everything in openmdao dir that isn't discarded is appended as a source package.
    for listing in os.listdir(os.path.join(docs_dir, "..")):
        if os.path.isdir(os.path.join("..", listing)):
            if listing not in IGNORE_LIST and listing not in packages:
                packages.append(listing)

    # begin writing the 'srcdocs/index.rst' file at mid  level.
    index_filename = os.path.join(doc_dir, "index.rst")
    index = open(index_filename, "w")
    if (doctype == "dev"):
        index.write(index_top_dev)
    else:
        index.write(index_top_usr)

    # auto-generate package header files (e.g. 'openmdao.core.rst')
    for package in packages:
        # a package is e.g. openmdao.core, that contains source files
        # a sub_package, is a src file, e.g. openmdao.core.component
        sub_packages = []
        package_filename = os.path.join(packages_dir,
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

            # specifically don't use os.path.join here.  Even windows wants the
            # stuff in the file to have fwd slashes.
            index.write("   packages/openmdao." + package + "\n")

            # make subpkg directory (e.g. srcdocs/packages/core) for ref sheets
            package_dir = os.path.join(packages_dir, package)
            os.mkdir(package_dir)

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
                    # specifically don't use os.path.join here.  Even windows wants the
                    # stuff in the file to have fwd slashes.
                    package_file.write("    " + package + "/" + sub_package + "\n")

                    # creates and writes out one reference sheet (e.g. core/component.rst)
                    ref_sheet_filename = os.path.join(package_dir, sub_package + ".rst")
                    ref_sheet = open(ref_sheet_filename, "w")

                    # get the meat of the ref sheet code done
                    filename = sub_package + ".py"
                    ref_sheet.write(".. index:: " + doctype + "_" + filename + "\n\n")
                    ref_sheet.write(".. _" + doctype + "_" + package_name + "." + filename + ":\n\n")
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


# generate docs, with private members, or without, based on doctype.
# type is passed in from the Makefile via the -t tags argument to sphinxbuild
if tags.has("dev"):
    doctype = "dev"
if tags.has("usr"):
    doctype = "usr"
else:
    doctype = None

if doctype:
    generate_docs(doctype)

# ------------------------begin monkeypatch-----------------------
# monkeypatch to make our docs say "Args" instead of "Parameters"
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
        out = self._str_indent(out, indent)
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

        # In creation of usr docs, remove private Attributes (beginning with '_')
        # with a crazy list comprehension
        if tags.has("usr"):
            self._parsed_data["Attributes"][:] = [att for att in self._parsed_data["Attributes"] if not att[0].startswith('_')]


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

# Do the actual patch switchover to these local versions
NumpyDocString.__init__ = __init__
SphinxDocString._str_options = _str_options
SphinxDocString._parse = _parse
SphinxDocString.__str__ = __str__
# --------------end monkeypatch---------------------

# --------------begin sphinx extension---------------------
# a short sphinx extension to take care of hyperlinking in docs
# where a syntax of <linktext> is employed.
import pkgutil
import inspect

# first, we will need a dict that contains full pathnames to every class.
# we construct that here, once, then use it for lookups in om_process_docstring
package = openmdao
om_classes = {}
for importer, modname, ispkg in pkgutil.walk_packages(path=package.__path__,
                                                      prefix=package.__name__+'.',
                                                      onerror=lambda x: None):
    if not ispkg:
        if 'docs' not in modname:
            module = importer.find_module(modname).load_module(modname)
            for classname, class_object in inspect.getmembers(module, inspect.isclass):
                if class_object.__module__.startswith("openmdao"):
                    om_classes[classname] = class_object.__module__ + "." + classname


def om_process_docstring(app, what, name, obj, options, lines):
    import re

    for i in range(len(lines)):
        # create a regex pattern to match <linktext>
        pat = r'(<.*?>)'
        # find all matches of the pattern in a line
        match = re.findall(pat, lines[i])
        if match:
            for ma in match:
                # strip off the angle brackets `<>`
                m = ma[1:-1]
                # if there's a dot in the pattern, it's a method
                # e.g. <classname.method_name>
                if '.' in m:
                    # need to grab the class name and method name separately
                    split_match = m.split('.')
                    justclass = split_match[0]  # class
                    justmeth = split_match[1]   # method
                    if justclass in om_classes:
                        classfullpath = om_classes[justclass]
                        # construct a link  :meth:`class.method <openmdao.core.class.method>`
                        link = ":meth:`" + m + " <" + classfullpath + "." + justmeth + ">`"
                        # replace the <link> text with the constructed line.
                        lines[i] = lines[i].replace(ma, link)
                    else:
                        # the class isn't in the class table!
                        print("WARNING: {} not found in dictionary of OpenMDAO methods".format(justclass))
                        # replace instances of <class> with just class in docstring (strip angle brackets)
                        lines[i] = lines[i].replace(ma, m)
                # otherwise, it's a class
                else:
                    if m in om_classes:
                        classfullpath = om_classes[m]
                        lines[i] = lines[i].replace(ma, ":class:`~"+classfullpath+"`")
                    else:
                        # the class isn't in the class table!
                        print("WARNING: {} not found in dictionary of OpenMDAO classes".format(m))
                        # replace instances of <class> with class in docstring (strip angle brackets)
                        lines[i] = lines[i].replace(ma, m)


# This is the crux of the extension--connecting an internal
# Sphinx event with our own custom function.
def setup(app):
    app.connect('autodoc-process-docstring', om_process_docstring)

#--------------end sphinx extension---------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('.'))

if (type == "usr"):
    absp = os.path.join('.', 'srcdocs', 'usr')
    sys.path.insert(0, os.path.abspath(absp))
elif (type == "dev"):
    absp = os.path.join('.', 'srcdocs', 'dev')
    sys.path.insert(0, os.path.abspath(absp))


# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = '1.5'

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

# autodoc_default_flags = ['members']

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
# version = openmdao.__version__
version = "2.0.0"
# The full version, including alpha/beta/rc tags.
# release = openmdao.__version__ + ' Alpha'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
if tags.has("usr"):
    exclude_patterns = ['_build', 'srcdocs/dev']

if tags.has("dev"):
    exclude_patterns = ['_build', 'srcdocs/usr']

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
