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


def write_tags(tag, content='', attrs=None, new_lines=False, indent=0):
    # Writes an HTML tag with element content and element attributes (given as a dictionary)
    line_sep = '\n' if new_lines else ''
    spaces = ' ' * indent
    template = '{spaces}<{tag} {attributes}>{ls}{content}{ls}</{tag}>\n'
    if attrs is None:
        attrs = {}
    attrs = ' '.join(['{}="{}"'.format(k, v) for k, v in iteritems(attrs)])
    return template.format(tag=tag, content=content, attributes=attrs, ls=line_sep, spaces=spaces)


def write_div(content='', attrs=None, indent=0):
    return write_tags('div', content, attrs, new_lines=False, indent=indent)


def write_style(content='', attrs=None, indent=0):
    default = {'type': "text/css"}
    if attrs is None:
        attrs = default
    else:
        attrs = default.update(attrs)
    return write_tags('style', content, attrs=attrs, new_lines=True, indent=indent)


def write_script(content='', attrs=None, indent=0):
    default = {'type': "text/javascript"}
    if attrs is None:
        attrs = default
    else:
        attrs = default.update(attrs)
    return write_tags('script', content, attrs=attrs, new_lines=True, indent=indent)


def read_files(filenames, directory, extension):
    # Reads files (based on filenames) from a directory with a given extension.
    libs = dict()
    for name in filenames:
        with open(os.path.join(directory, '.'.join([name, extension])), "r") as f:
            libs[name] = f.read()
    return libs

# Viewer API


def add_button(title, content='', indent=0):
    i = write_tags(tag='i', attrs={'class': content})
    return write_tags('button', content=i, attrs={'class': "myButton", 'title': title},
                      new_lines=True, indent=indent)


def add_dropdown(title, id_naming=None, options=None, button_content='', header=None,
                 dropdown_id=None, indent=0):
    button = add_button(title=title, content=button_content)
    if header is not None:
        items = write_tags(tag='span', attrs={'class': "fakeLink"}, content=header)
    else:
        items = ''

    if options is not None:
        for option in options:
            idx = "{}{}".format(id_naming, option)
            items += write_tags(tag='span', attrs={'class': "fakeLink", 'id': idx}, content=option)

    attrs = {'class': 'dropdown-content'}
    if dropdown_id is not None:
        attrs['id'] = dropdown_id
    menu = write_div(content=items, attrs=attrs)

    content = '\n'.join([button, menu])
    return write_div(content=content, attrs={'class': 'dropdown'}, indent=indent)
