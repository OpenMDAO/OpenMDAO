import json
import os

_DEFAULT_JSON_FILE = "xdsm.json"


def write_html(outfile, source_data):
    """
    Writes XDSMjs HTML output file, with style and script files embedded.

    The source data can be the name of a JSON file or a dictionary.
    If a JSON file name is provided, the file will be referenced in the HTML.
    If the input is a dictionary, it will be embedded.

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
    style_dir = code_dir

    # grab the libraries
    scripts = ''
    script_names = {'animation', 'controls', 'graph', 'labelizer', 'selectable', 'xdsm'}

    with open(os.path.join(build_dir, "xdsm.bundle.js"), "r") as f:
        xdsm_bundle = f.read()

    for name in script_names:
        with open(os.path.join(src_dir, "{}.js".format(name)), "r") as f:
            code = f.read()
            script = '<script type="text/javascript">\n{}\n</script>\n'.format(code)
            scripts += script

    with open(os.path.join(code_dir, "xdsm-main.js"), "r") as f:
        code = f.read()
        script = '<script type="text/javascript">\n{}\n</script>\n'.format(code)
        scripts += script

    if isinstance(source_data, str):
        data_name = source_data
        # replace file name
        xdsm_bundle = xdsm_bundle.replace(_DEFAULT_JSON_FILE, data_name)
    elif isinstance(source_data, dict):
        data_name = 'xdsm_data'
        tag = '<script id="{}" type="text/javascript">\nvar modelData = {}\n</script>\n'

        json_data = json.dumps(source_data)  # JSON formatted string
        script = tag.format(data_name, json_data)
        scripts += script
        # replace file name
        # FIXME this is wrong syntax now
        # TODO loading the JSON file should be replaced to use the modelData var or
        #  alternatively embed JSON script.
        xdsm_bundle = xdsm_bundle.replace('"xdsm.json",fetch("xdsm.json",void 0)', "modelData")
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
    with open(os.path.join(vis_dir, "index.html"), "r") as f:
        index = f.read()

    # put all style and JS into index
    index = index.replace('{{fontello_style}}', fontello_style)
    index = index.replace('{{xdsm_style}}', xdsm_style)
    index = index.replace('{{xdsm_bundle}}', xdsm_bundle)
    index = index.replace('{{scripts}}', scripts)

    with open(outfile, 'w') as f:
        f.write(index)


if __name__ == '__main__':
    # with JSON file name as input
    write_html(outfile='xdsmjs/xdsm_diagram.html', source_data="examples/idf.json")

    # with JSON data as input
    with open("XDSMjs/examples/idf.json") as f:
        data = json.load(f)
    write_html(outfile='xdsm_diagram_data_embedded.html', source_data=data)