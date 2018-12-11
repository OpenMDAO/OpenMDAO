import os

_DEFAULT_JSON_FILE = "xdsm.json"


def write_html(outfile='xdsmjs/xdsm_diagram.html', source_data="examples/idf.json"):

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
    else:
        data_name = 'xdsm_data'
        tag = '<script id="{}" type="text/javascript">\nvar modelData = {}\n</script>\n'

        json_data = json.dumps(source_data)  # JSON formatted string
        script = tag.format(data_name, json_data)
        scripts += script
        # replace file name
        xdsm_bundle = xdsm_bundle.replace('fetch("{}",void 0)'.format(_DEFAULT_JSON_FILE),
                                          "modelData")  # FIXME incorrect syntax

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
    write_html()  # with string JSON name
    # import json
    # with open("XDSMjs/examples/idf.json") as f:
    #     data = json.load(f)
    # write_html(source_data=data)