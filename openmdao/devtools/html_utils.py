"""
Functions to write HTML elements.
"""
import os

from six import iteritems


def head_and_body(head, body, attrs=None):
    # Wraps the head and body in tags
    doc_type = '<!doctype html>'
    head_elem = write_tags(tag='head', content=head, new_lines=True)
    body_elem = write_tags(tag='body', content=body, new_lines=True)
    content = '\n\n'.join([head_elem, body_elem])
    index = write_tags(tag='html', content=content, attrs=attrs, new_lines=True)
    return doc_type + '\n' + index


def write_tags(tag, content='', attrs=None, cls=None, new_lines=False, indent=0, **kwargs):
    # Writes an HTML tag with element content and element attributes (given as a dictionary)
    line_sep = '\n' if new_lines else ''
    spaces = ' ' * indent
    template = '{spaces}<{tag} {attributes}>{ls}{content}{ls}</{tag}>\n'
    if attrs is None:
        attrs = {}
    attrs.update(kwargs)
    if cls is not None:
        attrs['class'] = cls
    attrs = ' '.join(['{}="{}"'.format(k, v) for k, v in iteritems(attrs)])
    if isinstance(content, list):  # Convert iterable to string
        content = '\n'.join(content)
    return template.format(tag=tag, content=content, attributes=attrs, ls=line_sep, spaces=spaces)


def write_div(content='', attrs=None, cls=None, indent=0, **kwargs):
    return write_tags('div', content=content, attrs=attrs, cls=cls, new_lines=False,
                      indent=indent, **kwargs)


def write_style(content='', attrs=None, indent=0, **kwargs):
    default = {'type': "text/css"}
    if attrs is None:
        attrs = default
    else:
        attrs = default.update(attrs)
    return write_tags('style', content, attrs=attrs, new_lines=True, indent=indent, **kwargs)


def write_script(content='', attrs=None, indent=0, **kwargs):
    default = {'type': "text/javascript"}
    if attrs is None:
        attrs = default
    else:
        attrs = default.update(attrs)
    return write_tags('script', content, attrs=attrs, new_lines=True, indent=indent, **kwargs)


def read_files(filenames, directory, extension):
    # Reads files (based on filenames) from a directory with a given extension.
    libs = dict()
    for name in filenames:
        with open(os.path.join(directory, '.'.join([name, extension])), "r") as f:
            libs[name] = f.read()
    return libs

# Viewer API


def add_button(title, content='', button_id=None, indent=0, **kwargs):
    i = write_tags(tag='i', attrs={'class': content})
    attrs = {'title': title}
    if button_id:
        attrs['id'] = button_id
    return write_tags('button', cls="myButton", content=i, attrs=attrs,
                      new_lines=True, indent=indent, **kwargs)


def add_dropdown(title, id_naming=None, options=None, button_content='', header=None,
                 dropdown_id=None, indent=0, **kwargs):
    button = add_button(title=title, content=button_content)
    if header is not None:
        items = write_tags(tag='span', cls="fakeLink", content=header)
    else:
        items = ''

    if options is not None:
        for option in options:
            idx = "{}{}".format(id_naming, option)
            items += write_tags(tag='span', cls="fakeLink", attrs={'id': idx}, content=option)

    attrs = {'class': 'dropdown-content'}
    if dropdown_id is not None:
        attrs['id'] = dropdown_id
    menu = write_div(content=items, attrs=attrs)

    content = [button, menu]
    return write_div(content=content, cls='dropdown', indent=indent, **kwargs)


class ButtonGroup(object):
    """Button group, which consists of buttons and dropdowns."""

    def __init__(self, indent=0):
        self.items = []
        self.indent = indent

    def add_button(self, title, content='', button_id=None, **kwargs):
        button = add_button(title, content=content, button_id=button_id, indent=self.indent+4, **kwargs)
        self.items.append(button)

    def add_dropdown(self, title, id_naming=None, options=None, button_content='', header=None,
                     dropdown_id=None, **kwargs):
        dropdown = add_dropdown(self, title=title, id_naming=id_naming, options=options,
                                button_content=button_content, header=header,
                                dropdown_id=dropdown_id, indent=self.indent+4, **kwargs)
        self.items.append(dropdown)

    def write(self):
        """
        Outputs the HTML code.

        Returns
        -------
            str
        """
        content = '\n\n'.join(self.items)
        return write_div(content=content, cls="button-group")
