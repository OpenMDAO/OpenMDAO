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


def write_tags(tag, content, attrs=None, new_lines=False):
    # Writes an HTML tag with element content and element attributes (given as a dictionary)
    line_sep = '\n' if new_lines else ''
    template = '<{tag} {attributes}>{ls}{content}{ls}</{tag}>\n'
    if attrs is None:
        attrs = {}
    attrs = ' '.join(['{}="{}"'.format(k, v) for k, v in iteritems(attrs)])
    return template.format(tag=tag, content=content, attributes=attrs, ls=line_sep)


def write_div(content='', attrs=None):
    return write_tags('div', content, attrs, new_lines=False)


def write_style(content=''):
    return write_tags('style', content, attrs={'type': "text/css"}, new_lines=True)


def write_script(content='', attrs=None):
    return write_tags('script', content, attrs, new_lines=True)


def read_files(filenames, directory, extension):
    # Reads files (based on filenames) from a directory with a given extension.
    libs = dict()
    for name in filenames:
        with open(os.path.join(directory, '.'.join([name, extension])), "r") as f:
            libs[name] = f.read()
    return libs
