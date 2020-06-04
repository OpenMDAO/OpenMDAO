from numpydoc.docscrape_sphinx import SphinxDocString
from numpydoc.docscrape import NumpyDocString, Reader, ParseError
import textwrap

# start off running the monkeypatch to keep options/parameters
# usable in docstring for autodoc.


def __init__(self, docstring, config={}):
    """
    init
    """
    orig_docstring = docstring
    docstring = textwrap.dedent(docstring).split('\n')

    self._doc = Reader(docstring)
    self._parsed_data = {
        'Signature': '',
        'Summary': [''],
        'Extended Summary': [],
        'Parameters': [],
        'Options': [],
        'Returns': [],
        'Yields': [],
        'Raises': [],
        'Warns': [],
        'Other Parameters': [],
        'Attributes': [],
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

    # In creation of docs, remove private Attributes (beginning with '_')
    # with a crazy list comprehension
    self._parsed_data["Attributes"][:] = [att for att in self._parsed_data["Attributes"]
                                          if not att[0].startswith('_')]


def _parse(self):
    """
    parse
    """
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

        if section in ('Parameters', 'Options', 'Params', 'Returns', 'Yields', 'Raises',
                       'Warns', 'Other Parameters', 'Attributes',
                       'Methods'):
            self[section] = self._parse_param_list(content)
        elif section.startswith('.. index::'):
            self['index'] = self._parse_index(section, content)
        elif section == 'See Also':
            self['See Also'] = self._parse_see_also(content)
        else:
            self[section] = content


def __str__(self, indent=0, func_role="obj"):
    """
    our own __str__
    """
    out = []
    out += self._str_signature()
    out += self._str_index() + ['']
    out += self._str_summary()
    out += self._str_extended_summary()
    out += self._str_param_list('Parameters')
    out += self._str_options('Options')
    out += self._str_returns()
    for param_list in ('Other Parameters', 'Raises', 'Warns'):
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


def _str_options(self, name):
    """
    """
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
def do_monkeypatch():
    NumpyDocString.__init__ = __init__
    SphinxDocString._str_options = _str_options
    SphinxDocString._parse = _parse
    SphinxDocString.__str__ = __str__
