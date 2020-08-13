
import importlib

from docutils import nodes
from docutils.statemachine import ViewList

import sphinx
from docutils.parsers.rst import Directive
from sphinx.util.nodes import nested_parse_with_titles
from openmdao.utils.options_dictionary import OptionsDictionary, _undefined


class EmbedOptionsDirective(Directive):
    """
    EmbedOptionsDirective is a custom directive to allow an OptionsDictionary
    to be shown in a nice table form.  An example usage would look like this:

    .. embed-options::
        openmdao.solvers.linear.petsc_ksp
        PETScKrylov
        options

    The 3 arguments are the module path, the class name, and name of the options dictionary.

    What the above will do is replace the directive and its args with a list of options
    for the desired class.

    """

    required_arguments = 3
    optional_arguments = 0
    has_content = True

    def run(self):
        module_path, class_name, attribute_name = self.arguments

        mod = importlib.import_module(module_path)
        klass = getattr(mod, class_name)
        options = getattr(klass(), attribute_name)

        if not isinstance(options, OptionsDictionary):
            raise TypeError("Object '%s' is not an OptionsDictionary." % attribute_name)

        lines = ViewList()

        n = 0
        for line in options.__rst__():
            lines.append(line, "options table", n)
            n += 1

        # Note applicable to System, Solver and Driver 'options', but not to 'recording_options'
        if attribute_name != 'recording_options':
            lines.append("", "options table", n+1)  # Blank line required after table.

        # Create a node.
        node = nodes.section()
        node.document = self.state.document

        # Parse the rst.
        nested_parse_with_titles(self.state, lines, node)

        # And return the result.
        return node.children


def setup(app):
    """add custom directive into Sphinx so that it is found during document parsing"""
    app.add_directive('embed-options', EmbedOptionsDirective)

    return {'version': sphinx.__display_version__, 'parallel_read_safe': True}
