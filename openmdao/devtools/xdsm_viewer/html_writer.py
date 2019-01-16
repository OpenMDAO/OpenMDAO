"""
HTML file writing to create (semi)standalone XDSMjs output file.
"""

import json
import os

from six import iteritems

_DEFAULT_JSON_FILE = "xdsm.json"
_HTML_TEMPLATE = "index.html"


def write_html(outfile, source_data):
    """
    Writes XDSMjs HTML output file, with style and script files embedded.

    The source data can be the name of a JSON file or a dictionary.
    If a JSON file name is provided, the file will be referenced in the HTML.
    If the input is a dictionary, it will be embedded.

    Parameters
    ----------
    outfile : str
        Output HTML file
    source_data : str or dict
        Output HTML file
    """

    # directories
    main_dir = os.path.dirname(os.path.abspath(__file__))
    code_dir = os.path.join(main_dir, 'XDSMjs')
    src_dir = os.path.join(code_dir, "src")
    build_dir = os.path.join(code_dir, "build")
    vis_dir = os.path.join(main_dir, "visualization")
    style_dir = code_dir  # CSS

    # grab the libraries
    scripts = ''
    script_names = {'animation', 'controls', 'graph', 'labelizer', 'selectable', 'xdsm',
                    'xdsm-factory'}

    with open(os.path.join(build_dir, "xdsm.bundle.js"), "r") as f:
        code = f.read()
        xdsm_bundle = _write_script(code, {'type': 'text/javascript'})
    # grab the scripts
    for name in script_names:
        with open(os.path.join(src_dir, "{}.js".format(name)), "r") as f:
            code = f.read()
            script = _write_script(code, {'type': 'text/javascript'})
            scripts += script

    xdsm_attrs = {'class': 'xdsm'}
    # grab the data
    if isinstance(source_data, str):
        # Add name of the data file
        xdsm_attrs['data-mdo-file'] = source_data
    elif isinstance(source_data, dict):
        msg = ('The option to embed the data is not implemented yet. '
               'Use a JSON file name as input instead.')
        raise NotImplementedError(msg)
        # tag = script_tag.format('var modelData = {}')
        #
        # json_data = json.dumps(source_data)  # JSON formatted string
        # script = tag.format(json_data)
        # scripts += script
        # # replace file name
        # # FIXME this is wrong syntax now
        # # TODO loading the JSON file should be replaced to use the modelData var or
        # #  alternatively embed JSON script.
        # xdsm_bundle = xdsm_bundle.replace('"xdsm.json",fetch("xdsm.json",void 0)', "modelData")
    else:
        msg = ('Invalid data type for source data: {} \n'
               'The source data should be a JSON file name or a dictionary.')
        raise ValueError(msg.format(type(source_data)))

    # grab the style
    with open(os.path.join(style_dir, "fontello.css"), "r") as f:
        fontello_style = f.read()
    with open(os.path.join(style_dir, "xdsm.css"), "r") as f:
        xdsm_style = f.read()

    # grab the index.html
    with open(os.path.join(vis_dir, _HTML_TEMPLATE), "r") as f:
        index = f.read()

    # put all style and JS into index
    toolbar_div = _write_div(attrs={'class': 'xdsm-toolbar'})
    xdsm_div = _write_div(attrs=xdsm_attrs)
    index = index.replace('{{body}}', toolbar_div + '\n' + xdsm_div)
    index = index.replace('{{fontello_style}}', fontello_style)
    index = index.replace('{{xdsm_style}}', xdsm_style)
    index = index.replace('{{xdsm_bundle}}', xdsm_bundle)
    index = index.replace('{{scripts}}', scripts)

    # Embed style, scripts and data to HTML
    with open(outfile, 'w') as f:
        f.write(index)


def _write_tags(tag, content, attrs, new_lines=False):
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


if __name__ == '__main__':
    # with JSON file name as input
    write_html(outfile='xdsmjs/xdsm_diagram.html', source_data="examples/idf.json")

    # # with JSON data as input
    # with open("XDSMjs/examples/idf.json") as f:
    #     data = json.load(f)
    # write_html(outfile='xdsm_diagram_data_embedded.html', source_data=data)
