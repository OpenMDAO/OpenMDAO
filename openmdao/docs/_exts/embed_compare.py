""" Sphinx directive for a side by side code comparison."""

from docutils import nodes

import sphinx
from docutils.parsers.rst import Directive
from openmdao.docs._utils.docutil import get_source_code


class ContentContainerDirective(Directive):
    """
    Just for having an outer div.

    Relevant CSS: rosetta_outer
    """

    has_content = True
    optional_arguments = 1

    def run(self):
        self.assert_has_content()
        text = '\n'.join(self.content)
        node = nodes.container(text)
        node['classes'].append('rosetta_outer')

        if self.arguments and self.arguments[0]:
            node['classes'].append(self.arguments[0])

        self.add_name(node)
        self.state.nested_parse(self.content, self.content_offset, node)
        return [node]


class EmbedCompareDirective(Directive):
    """
    EmbedCompareDirective is a custom directive to allow blocks of
    python code to be shown side by side to compare the new API with the old API. An
    exmple looks like this:

    .. embed-compare::
        openmdao.test.whatever.method
        optional text for searching for the first line
        optional text for searching for the end line
        optional style

      Old OpenMDAO lines of code go here.

    What the above will do is replace the directive and its args with the block of code
    containing the class for method1 on the left and the class for method2 on the right.

    For optional styles, use 'style2' to use the alternate CSS style that has a light background on
    both sides instead of red and green. Use 'no_compare' for straight code embedding without the
    side-by-side comparison. (This is for pasting fragments of pre-tested code from a test.)

    Relevant CSS: rosetta_left and rosetta_right
    """

    # must have at least one directive for this to work
    required_arguments = 1
    optional_arguments = 3
    has_content = True

    def run(self):
        arg = self.arguments
        compare = True

        # create a list of document nodes to return
        doc_nodes = []

        # Choose style
        left_style = 'rosetta_left'
        right_style = 'rosetta_right'
        if len(arg) == 4:
            if arg[3] == 'style2':
                left_style = 'rosetta_left2'
                right_style = 'rosetta_right2'
            elif arg[3] == 'no_compare':
                compare = False

        # LEFT side = Old OpenMDAO
        if compare:
            text = '\n'.join(self.content)
            left_body = nodes.literal_block(text, text)
            left_body['language'] = 'python'
            left_body['classes'].append(left_style)

        # for RIGHT side, get the code block, and reduce it if requested
        right_method = arg[0]
        text, _, _, _, _ = get_source_code(right_method)
        if len(arg) >= 3:
            start_txt = arg[1]
            end_txt = arg[2]
            lines = text.split('\n')

            istart = 0
            for j, line in enumerate(lines):
                if start_txt in line:
                    istart = j
                    break

            lines = lines[istart:]
            iend = len(lines)
            for j, line in enumerate(lines):
                if end_txt in line:
                    iend = j+1
                    break

            lines = lines[:iend]

            # Remove the check suppression.
            for j, line in enumerate(lines):
                if "prob.setup(check=False" in line:
                    lines[j] = lines[j].replace('check=False, ', '')
                    lines[j] = lines[j].replace('check=False', '')

            # prune whitespace down to match first line
            while lines[0].startswith('    '):
                lines = [line[4:] for line in lines]

            text = '\n'.join(lines)

        # RIGHT side = Current OpenMDAO
        right_body = nodes.literal_block(text, text)
        right_body['language'] = 'python'
        if compare:
            right_body['classes'].append(right_style)

        if compare:
            doc_nodes.append(left_body)
        doc_nodes.append(right_body)

        return doc_nodes


def setup(app):
    """add custom directive into Sphinx so that it is found during document parsing"""
    app.add_directive('content-container',  ContentContainerDirective)
    app.add_directive('embed-compare', EmbedCompareDirective)

    return {'version': sphinx.__display_version__, 'parallel_read_safe': True}
