"""
HTML file writing to create (semi)standalone XDSMjs output file.
"""

import json
import os

from six import iteritems, itervalues

_DEFAULT_JSON_FILE = "xdsm.json"  # Used as default name if data is not embedded
_CHAR_SET = "utf-8"  # HTML character set


def write_html(outfile, source_data=None, data_file=None, embeddable=False):
    """
    Writes XDSMjs HTML output file, with style and script files embedded.

    The source data can be the name of a JSON file or a dictionary.
    If a JSON file name is provided, the file will be referenced in the HTML.
    If the input is a dictionary, it will be embedded.

    If both data file and source data are given, data file is

    Parameters
    ----------
    outfile : str
        Output HTML file
    source_data : str or dict or None
        XDSM data in a dictionary or string
    data_file : str or None
        Output HTML file
    embeddable : bool, optional
        If True, gives a single HTML file that doesn't have the <html>, <DOCTYPE>, <body>
        and <head> tags. If False, gives a single, standalone HTML file for viewing.
    """

    # directories
    main_dir = os.path.dirname(os.path.abspath(__file__))
    code_dir = os.path.join(main_dir, 'XDSMjs')
    build_dir = os.path.join(code_dir, "build")
    style_dir = code_dir  # CSS

    with open(os.path.join(build_dir, "xdsm.bundle.js"), "r") as f:
        code = f.read()
        xdsm_bundle = _write_script(code, {'type': 'text/javascript'})

    xdsm_attrs = {'class': 'xdsm'}
    # grab the data
    if data_file is not None:
        # Add name of the data file
        xdsm_attrs['data-mdo-file'] = data_file
    elif source_data is not None:
        if isinstance(source_data, (dict, str)):
            data_str = str(source_data)  # dictionary converted to string
        else:
            msg = ('Invalid data type for source data: {} \n'
                   'The source data should be a JSON file name or a dictionary.')
            raise ValueError(msg.format(type(source_data)))

        # Replace quote marks for the HTML syntax
        for i in ('u"', "u'", '"', "'"):  # quote marks and unicode prefixes
            data_str = data_str.replace(i, r'&quot;')
        xdsm_attrs['data-mdo'] = data_str
    else:  # both source data and data file name are None
        msg = 'Specify either "source_data" or "data_file".'
        raise ValueError(msg.format(type(source_data)))

    # grab the style
    styles = _read_files(('fontello', 'xdsm'), style_dir, 'css')
    styles_elem = _write_tags(tag='style',
                              content='\n\n'.join(itervalues(styles)),
                              attrs={'type': "text/css"},
                              new_lines=True)

    # put all style and JS into index
    toolbar_div = _write_div(attrs={'class': 'xdsm-toolbar'})
    xdsm_div = _write_div(attrs=xdsm_attrs)
    body = '\n\n'.join([toolbar_div, xdsm_div])

    if embeddable:
        index = '\n\n'.join([styles_elem, xdsm_bundle, body])
    else:
        meta = '<meta charset="{}">'.format(_CHAR_SET)

        head = '\n\n'.join([meta, styles_elem, xdsm_bundle])

        index = _head_and_body(head, body, attrs={'class': "js", 'lang': ""})

    # Embed style, scripts and data to HTML
    with open(outfile, 'w') as f:
        f.write(index)


def _head_and_body(head, body, attrs):
    # Wraps the head and body in tags
    doc_type = '<!doctype html>'
    head_elem = _write_tags(tag='head', content=head, new_lines=True)
    body_elem = _write_tags(tag='body', content=body, new_lines=True)
    content = '\n\n'.join([head_elem, body_elem])
    index = _write_tags(tag='html', content=content, attrs=attrs, new_lines=True)
    return doc_type + '\n' + index


def _write_tags(tag, content, attrs=None, new_lines=False):
    # Writes an HTML tag with element content and element attributes (given as a dictionary)
    line_sep = '\n' if new_lines else ''
    template = '<{tag} {attributes}>{ls}{content}{ls}</{tag}>\n'
    if attrs is None:
        attrs = {}
    attrs = ' '.join(['{}="{}"'.format(k, v) for k, v in iteritems(attrs)])
    return template.format(tag=tag, content=content, attributes=attrs, ls=line_sep)


def _write_div(content='', attrs=None):
    return _write_tags('div', content, attrs, new_lines=False)


def _write_script(content='', attrs=None):
    return _write_tags('script', content, attrs, new_lines=True)


def _read_files(filenames, directory, extension):
    # Reads files (based on filenames) from a directory with a given extension.
    libs = dict()
    for name in filenames:
        with open(os.path.join(directory, '.'.join([name, extension])), "r") as f:
            libs[name] = f.read()
    return libs


if __name__ == '__main__':
    # with JSON file name as input
    write_html(outfile='xdsmjs/xdsm_diagram.html', source_data="examples/idf.json")

    # with JSON data as input
    with open("XDSMjs/examples/idf.json") as f:
        data = json.load(f)
    write_html(outfile='xdsm_diagram_data_embedded.html', source_data=data)
