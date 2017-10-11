from six import iteritems
from docutils import nodes
from sphinx.util.compat import Directive


class EmbedOptionsDirective(Directive):
    """
    EmbedOptionsDirective is a custom directive to allow an OptionsDictionary
    to be shown in a nice table form.  An example usage would look like this:

    .. embed-options::
        openmdao.solvers.linear.petsc_ksp , PetscKSP , options

    The 3 arguments are the module, the class path, and name of the options dictionary.

    What the above will do is replace the directive and its args with the block of code
    for the class or method desired.

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
        embed_num_indent = 3
        doc_nodes = []

        exec('from {} import {}'.format(module_path, class_name), globals())
        exec('obj = {}()'.format(class_name), globals())

        options = getattr(obj, attribute_name)

        outputs = []
        for option_name, option_data in iteritems(options._dict):
            name = option_name
            default = option_data['value']
            values = option_data['values']
            types = option_data['type_']
            desc = option_data['desc']

            if types is not None:
                if not isinstance(types, (tuple, list)):
                    types = (types,)

                types = [type_.__name__ for type_ in types]

            if values is not None:
                if not isinstance(values, (tuple, list)):
                    values = (values,)

                values = [value for value in values]

            outputs.append([name, default, values, types, desc])

        lines = []
        lines.append(' ' * embed_num_indent
            + '.. list-table:: List of options\n')
        lines.append(' ' * embed_num_indent
            + ' ' * 2 + ':header-rows: 1\n')
        lines.append(' ' * embed_num_indent
            + ' ' * 2 + ':widths: 15, 10, 20, 20, 30\n')
        lines.append(' ' * embed_num_indent
            + ' ' * 2 + ':stub-columns: 0\n')
        lines.append('\n')
        lines.append(' ' * embed_num_indent
            + ' ' * 2 + '*  -  Option\n')
        lines.append(' ' * embed_num_indent
            + ' ' * 2 + '   -  Default\n')
        lines.append(' ' * embed_num_indent
            + ' ' * 2 + '   -  Acceptable values\n')
        lines.append(' ' * embed_num_indent
            + ' ' * 2 + '   -  Acceptable types\n')
        lines.append(' ' * embed_num_indent
            + ' ' * 2 + '   -  Description\n')

        for output in outputs:
            for entry in [output[0]]:
                lines.append(' ' * embed_num_indent
                    + ' ' * 2 + '*  -  %s\n' % entry)
            for entry in output[1:]:
                lines.append(' ' * embed_num_indent
                    + ' ' * 2 + '   -  %s\n' % entry)

        table = nodes.Text(lines, lines)
        doc_nodes.append(table)
        return doc_nodes

def setup(app):
    """add custom directive into Sphinx so that it is found during document parsing"""
    app.add_directive('embed-options', EmbedOptionsDirective)
