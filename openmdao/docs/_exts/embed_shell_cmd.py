"""
Sphinx directive to embed shell command output in the docs.

The shell command is executed and the output is captured.
"""

import sys
import os
from docutils import nodes
from docutils.parsers.rst.directives import unchanged

import subprocess

import sphinx
from docutils.parsers.rst import Directive
from sphinx.writers.html import HTMLTranslator
from sphinx.writers.html5 import HTML5Translator
from sphinx.errors import SphinxError


if sys.version_info[0] == 2:
    import cgi as cgiesc
else:
    import html as cgiesc


class failed_node(nodes.Element):
    pass


def visit_failed_node(self, node):
    pass


def depart_failed_node(self, node):
    if not isinstance(self, (HTMLTranslator, HTML5Translator)):
        self.body.append("output only available for HTML\n")
        return

    html = """
    <div class="cell border-box-sizing code_cell rendered">
       <div class="output">
          <div class="inner_cell">
             <div class="failed"><pre>{}</pre></div>
          </div>
       </div>
    </div>""".format(node["text"])
    self.body.append(html)


class cmd_node(nodes.Element):
    pass


def visit_cmd_node(self, node):
    pass


def depart_cmd_node(self, node):
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


class EmbedShellCmdDirective(Directive):
    """
    EmbedShellCmdDirective is a custom directive to allow a shell command and the result
    of running it to be shown in feature docs.
    An example usage would look like this:

    .. embed-shell-cmd::
        :cmd: ls -ltr

    What the above will do is replace the directive and its args with the shell command,
    run the command, and show the resulting output.

    """

    # must have at least one arg (embedded test) for this to work
    required_arguments = 0
    optional_arguments = 0
    has_content = False

    option_spec = {
        'cmd': unchanged,        # shell command to execute
        'dir': unchanged,        # working dir
        'show_cmd': unchanged,   # set this to make the shell command visible
        'stderr': unchanged      # set this to include stderr contents with the output
    }

    def run(self):
        """
        Create a list of document nodes to return.
        """
        if 'cmd' in self.options:
            cmdstr = self.options['cmd']
            cmd = cmdstr.split()
        else:
            raise SphinxError("'cmd' is not defined for embed-shell-cmd.")

        startdir = os.getcwd()

        if 'dir' in self.options:
            workdir = os.path.abspath(os.path.expandvars(os.path.expanduser(self.options['dir'])))
        else:
            workdir = os.getcwd()

        if 'stderr' in self.options:
            stderr = subprocess.STDOUT
        else:
            stderr = None

        os.chdir(workdir)

        try:
            output = subprocess.check_output(cmd, stderr=subprocess.STDOUT,env=os.environ).decode('utf-8', 'ignore')
        except subprocess.CalledProcessError as err:
            # Failed cases raised as a Directive warning (level 2 in docutils).
            # This way, the sphinx build does not terminate if, for example, you are building on
            # an environment where mpi or pyoptsparse are missing.
            msg = "Running of embedded shell command '{}' in docs failed. Output was: \n{}" \
                  .format(cmdstr, err.output.decode('utf-8'))
            raise self.directive_error(2, msg)
        except Exception as err:
            msg = "Running of embedded shell command '{}' in docs failed. Output was: \n{}" \
                  .format(cmdstr, err)
            raise self.directive_error(2, msg)
        finally:
            os.chdir(startdir)

        output = cgiesc.escape(output)

        show = True
        if 'show_cmd' in self.options:
            show = self.options['show_cmd'].lower().strip() == 'true'

        if show:
            input_node = nodes.literal_block(cmdstr, cmdstr)
            input_node['language'] = 'none'

        output_node = cmd_node(text=output)

        if show:
            return [input_node, output_node]
        else:
            return [output_node]


def setup(app):
    """add custom directive into Sphinx so that it is found during document parsing"""
    app.add_directive('embed-shell-cmd', EmbedShellCmdDirective)
    app.add_node(failed_node, html=(visit_failed_node, depart_failed_node))
    app.add_node(cmd_node, html=(visit_cmd_node, depart_cmd_node))

    return {'version': sphinx.__display_version__, 'parallel_read_safe': True}
