from docutils import nodes
from docutils.statemachine import ViewList
from docutils.parsers.rst import Directive
import subprocess
import sphinx
from sphinx.util.nodes import nested_parse_with_titles
import os.path


class EmbedN2Directive(Directive):
    """
    EmbedN2Directive is a custom directive to build and embed an N2 diagram into docs
    An example usage would look like this:

    .. embed-n2::
        ../../examples/model.py

    What the above will do is replace the directive and its arg with an N2 diagram.

    The one required argument is the model file to be diagrammed.
    Optional arguments are numerical width and height of the embedded object, and
    "toolbar" if the toolbar should be visible by default.

    Example with width of 1500, height of 800, and toolbar displayed:

    .. embed-n2:
        ../../examples/model.py
        1500
        800
        toolbar

    """

    required_arguments = 1
    optional_arguments = 3
    has_content = True

    def run(self):
        path_to_model = self.arguments[0]
        n2_dims = [ 1200, 700 ]
        show_toolbar = False

        if len(self.arguments) > 1 and self.arguments[1]:
            n2_dim_idx = 0
            for idx in range(1, len(self.arguments)):
                if self.arguments[idx] == "toolbar":
                    show_toolbar = True
                else:
                    n2_dims[n2_dim_idx] = self.arguments[idx]
                    n2_dim_idx = 1

        np = os.path.normpath(os.path.join(os.getcwd(), path_to_model))

        # check that the file exists
        if not os.path.isfile(np):
            raise IOError('File does not exist({0})'.format(np))
        
        # Generate N2 files into the target_dir. Those files are later copied
        # into the top of the HTML hierarchy, so the HTML doc file needs a
        # relative path to them.
        target_dir = os.path.join(os.getcwd(), "_n2html")

        rel_dir = os.path.relpath(os.getcwd(),
                                  os.path.dirname(self.state.document.settings._source))
        html_base_name = os.path.basename(path_to_model).split('.')[0] + "_n2.html"
        html_name = os.path.join(target_dir, html_base_name)
        html_rel_name = os.path.join(rel_dir, html_base_name)
        if show_toolbar:
            html_rel_name += '#toolbar'

        cmd = subprocess.Popen(
            ['openmdao', 'n2', np, '--no_browser', '--embed', '-o' + html_name])
        cmd_out, cmd_err = cmd.communicate()

        rst = ViewList()

        # Add the content one line at a time.
        # Second argument is the filename to report in any warnings
        # or errors, third argument is the line number.
        env = self.state.document.settings.env
        docname = env.doc2path(env.docname)

        object_tag = (
            "<iframe width='" + str(n2_dims[0]) + "'"
            " height='" + str(n2_dims[1]) + "'"
            " style='border: 1px solid lightgray; resize: both;'"
            " src='" + html_rel_name + "'></iframe>"
        )

        rst.append(".. raw:: html", docname, self.lineno)
        rst.append("", docname, self.lineno)  # leave an empty line
        rst.append("    %s" % object_tag, docname, self.lineno)

        # Create a node.
        node = nodes.section()

        # Parse the rst.
        nested_parse_with_titles(self.state, rst, node)

        # And return the result.
        return node.children


def setup(app):
    """add custom directive into Sphinx so that it is found during document parsing"""
    app.add_directive('embed-n2', EmbedN2Directive)

    return {'version': sphinx.__display_version__, 'parallel_read_safe': True}
