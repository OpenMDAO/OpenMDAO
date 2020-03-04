
import sys
import importlib

from docutils import nodes

import sphinx
from docutils.parsers.rst import Directive

from sphinx.writers.html import HTMLTranslator
from sphinx.writers.html5 import HTML5Translator
from sphinx.errors import SphinxError


class bibtex_node(nodes.Element):
    pass


def visit_bibtex_node(self, node):
    pass


def depart_bibtex_node(self, node):
    """
    This function creates the formatting that sets up the look of the blocks.
    The look of the formatting is controlled by _theme/static/style.css
    """
    if not isinstance(self, (HTMLTranslator, HTML5Translator)):
        self.body.append("output only available for HTML\n")
        return

    html = """
    <div class="cell border-box-sizing code_cell rendered">
       <div class="output_area"><pre>{}</pre></div>
    </div>""".format(node["text"])

    self.body.append(html)


class EmbedBibtexDirective(Directive):
    """
    EmbedBibtexDirective is a custom directive to allow a Bibtex citation to be embedded.

    .. embed-bibtex::
        openmdao.solvers.linear.petsc_ksp
        PETScKrylov


    The 2 arguments are the module path and the class name.

    What the above will do is replace the directive and its args with the Bibtex citation
    for the class.

    """

    required_arguments = 2
    optional_arguments = 0
    has_content = True

    def run(self):
        module_path, class_name = self.arguments
        mod = importlib.import_module(module_path)
        obj = getattr(mod, class_name)()

        if not hasattr(obj, 'cite') or not obj.cite:
            raise SphinxError("Couldn't find 'cite' in class '%s'" % class_name)

        return [bibtex_node(text=obj.cite)]


def setup(app):
    """add custom directive into Sphinx so that it is found during document parsing"""
    app.add_directive('embed-bibtex', EmbedBibtexDirective)
    app.add_node(bibtex_node, html=(visit_bibtex_node, depart_bibtex_node))

    return {'version': sphinx.__display_version__, 'parallel_read_safe': True}
