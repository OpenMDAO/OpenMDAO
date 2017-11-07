from six import iteritems
from docutils import nodes
from docutils.statemachine import ViewList

import sphinx

from sphinx.util.compat import Directive
from sphinx.util.nodes import nested_parse_with_titles


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
        if self.arguments and self.arguments[0]:
            module_path = self.arguments[0]
        if self.arguments and self.arguments[1]:
            class_name = self.arguments[1]
        if self.arguments and self.arguments[2]:
            attribute_name = self.arguments[2]

        exec('from {} import {}'.format(module_path, class_name), globals())
        exec('obj = {}()'.format(class_name), globals())

        options = getattr(obj, attribute_name)

        outputs = []
        for option_name, option_data in sorted(iteritems(options._dict)):
            name = option_name
            default = option_data['value']
            values = option_data['values']
            types = option_data['types']
            desc = option_data['desc']

            if types is None:
                types = "N/A"

            elif types is not None:
                if not isinstance(types, (tuple, list)):
                    types = (types,)

                types = [type_.__name__ for type_ in types]

            if values is None:
                values = "N/A"

            elif values is not None:
                if not isinstance(values, (tuple, list)):
                    values = (values,)

                values = [value for value in values]

            outputs.append([name, default, values, types, desc])

        lines = ViewList()

        col_heads = ['Option', 'Default', 'Acceptable Values', 'Acceptable Types', 'Description']

        max_sizes = {}
        for j, col in enumerate(col_heads):
            max_sizes[j] = len(col)

        for output in outputs:
            for j, item in enumerate(output):
                length = len(str(item))
                if max_sizes[j] < length:
                    max_sizes[j] = length

        header = ""
        titles = ""
        for key, val in iteritems(max_sizes):
            header += '=' * val + ' '

        for j, head in enumerate(col_heads):
            titles += "%s " % head
            size = max_sizes[j]
            space = size - len(head)
            if space > 0:
                titles += space*' '

        lines.append(header, "options table", 1)
        lines.append(titles, "options table", 2)
        lines.append(header, "options table", 3)

        n = 3
        for output in outputs:
            line = ""
            for j, item in enumerate(output):
                line += "%s " % str(item)
                size = max_sizes[j]
                space = size - len(str(item))
                if space > 0:
                    line += space*' '

            lines.append(line, "options table", n)
            n += 1

        lines.append(header, "options table", n)

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
