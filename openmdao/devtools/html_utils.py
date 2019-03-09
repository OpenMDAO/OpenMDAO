"""
Functions to write HTML elements.
"""
import os

from six import iteritems, itervalues

_IND = 4  # indentation (spaces)


def head_and_body(head, body, attrs=None):
    # Wraps the head and body in tags
    doc_type = '<!doctype html>'
    head_elem = write_tags(tag='head', content=head, new_lines=True)
    body_elem = write_tags(tag='body', content=body, new_lines=True)
    content = '\n\n'.join([head_elem, body_elem])
    index = write_tags(tag='html', content=content, attrs=attrs, new_lines=True)
    return doc_type + '\n' + index


def write_tags(tag, content='', attrs=None, cls=None, new_lines=False, indent=0, **kwargs):
    """
    Writes an HTML element enclosed in tags.

    Parameters
    ----------
    tag : str
        Name of the tag.
    content : str or list(str)
        This goes into the body of the element.
    attrs : dict
        Attributes of the element.
    cls : str
        The "class" attribute of the element.
    new_lines : str
        Make new line after tags.
    indent : int
        Indentation expressed in spaces.
    kwargs
        Alternative way to add element attributes. Use with attention, can overwrite some in-bult
        python names as "class" or "id" if misused.

    Returns
    -------

    """
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
    """
    Writes an HTML div.

    Parameters
    ----------
    content : str or list(str)
        This goes into the body of the element.
    attrs : dict
        Attributes of the element.
    cls : str
        The "class" attribute of the element.
    indent : int
        Indentation expressed in spaces.
    kwargs
        Alternative way to add element attributes. Use with attention, can overwrite some in-bult
        python names as "class" or "id" if misused.

    Returns
    -------
        str
    """
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


def _p(content):
    return write_tags(tag='p', content=content)


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
                 dropdown_id=None, indent=0, option_formatter=None, **kwargs):
    button = add_button(title=title, content=button_content)
    if header is None:
        header = title
    items = write_tags(tag='span', cls="fakeLink", content=header)

    if options is not None:
        for option in options:
            if option_formatter is not None:
                option = option_formatter(option)
            idx = "{}{}".format(id_naming, option)
            items += write_tags(tag='span', cls="fakeLink", attrs={'id': idx}, content=option)

    attrs = {'class': 'dropdown-content'}
    if dropdown_id is not None:
        attrs['id'] = dropdown_id
    menu = write_div(content=items, attrs=attrs)

    content = [button, menu]
    return write_div(content=content, cls='dropdown', indent=indent, **kwargs)


def add_help(txt, header='Instructions', footer=''):
    header_txt = write_tags(tag='span', cls='close', content='&times;', attrs={'id': "idSpanModalClose"})
    header_txt += header
    head = write_div(content=header_txt, cls="modal-header")
    foot = write_div(content=footer, cls="modal-footer")
    body = write_div(content=_p(txt), cls="modal-body")
    modal_content = write_div(content=[head, body, foot], cls="modal-content")
    return write_div(content=modal_content, cls="modal", attrs={'id': "myModal"})


def add_title(txt):
    title = write_tags(tag='h1', content=txt)
    return write_div(content=title, attrs={'id': "maintitle", 'style': "text-align: center"})


class UIElement(object):
    """Abstract class for user interface elements."""

    def __init__(self, indent=0):
        self.items = []
        self.indent = indent


class ButtonGroup(UIElement):
    """
    Button group, which consists of buttons and dropdowns. Write it to get the HTML for the
    button group.
    """

    def add_button(self, title, content='', button_id=None, **kwargs):
        """
        Adds a button to the button group.

        Parameters
        ----------
        title : str
            Name to be shown.
        content : str
            The content of the element.
        button_id : str
            ID.
        kwargs : dict
            Attributes passed to the button element.
        Returns
        -------
            str
        """
        button = add_button(title, content=content, button_id=button_id, indent=self.indent+_IND,
                            **kwargs)
        self.items.append(button)
        return button

    def add_dropdown(self, title, id_naming=None, options=None, button_content='', header=None,
                     dropdown_id=None, option_formatter=None, **kwargs):
        """
        Adds a dropdown to the button group.

        Parameters
        ----------
        title : str
            Name to be shown.
        id_naming : str
            ID of an item will be id_naming + option
        options : list(str)
            Items of the dropdown.
        button_content : str
            Content of the button.
        header : str
            First item in the dropdown. Defaults to the title.
        dropdown_id : str
            ID.
        option_formatter : None or callable
            An optional text formatter for the dropdown items. Called with one item.
        kwargs : dict
            Attributes passed to the dropdown element.
        """
        dropdown = add_dropdown(title=title, id_naming=id_naming, options=options,
                                button_content=button_content, header=header,
                                dropdown_id=dropdown_id, indent=self.indent+_IND,
                                option_formatter=option_formatter, **kwargs)
        self.items.append(dropdown)
        return dropdown

    def write(self):
        """
        Outputs the HTML code.

        Returns
        -------
            str
        """
        content = '\n\n'.join(self.items)
        return write_div(content=content, cls="button-group")


class Toolbar(UIElement):
    """
    A toolbar consists of button groups. Add button groups, and write it to get the HTML for the
    toolbar.
    """

    def add_button_group(self):
        button_group = ButtonGroup(indent=self.indent+4)
        self.items.append(button_group)
        return button_group

    def write(self):
        """
        Outputs the HTML code.

        Returns
        -------
            str
        """
        content = '\n\n'.join([item.write() for item in self.items])
        return write_div(content=content, cls="toolbar", attrs={'id': "toolbarDiv"})


class TemplateWriter(object):
    """
    Opens an HTML template files, text can be inserted into the template, and writes  anew HTML
    file with the replacements.
    """

    def __init__(self, filename, embeddable=False, title=None, styles=None):
        self.filename = filename
        # Load template
        with open(self.filename, "r") as f:
            self.template = template = f.read()

        if styles is not None:
            style_elems = '\n\n'.join([write_style(content=s) for s in itervalues(styles)])

            if embeddable:
                self.template = '\n\n'.join([style_elems, template])
            else:
                meta = '<meta http-equiv="Content-Type" content="text/html; charset=UTF-8">'
                head = '\n\n'.join([meta, style_elems])  # Write styles to head
                self.template = head_and_body(head=head, body=template)

        if title is not None:
            self._replace('{{title}}', add_title(title))

    def _replace(self, ref, txt):
        # Replace a reference in the template file with a text
        self.template = self.template.replace(ref, txt)

    def insert(self, ref, txt):
        self._replace(ref=ref, txt=txt)

    def write(self, outfile):
        with open(outfile, 'w') as f:  # write output file
            f.write(self.template)


class DiagramWriter(TemplateWriter):
    """
    An HTML diagram writer. The diagram has a toolbar, which can be edited by adding
    button groups, dropdowns, buttons, etc. to this class.
    """

    def __init__(self, filename, embeddable=False, title=None, styles=None):
        super(DiagramWriter, self).__init__(filename=filename, embeddable=embeddable, title=title,
                                            styles=styles)
        self.toolbar = Toolbar()
        self.help = None

    def add_help(self, txt, header='Instructions', footer=''):
        """
        Adds a modal with instructions.

        Parameters
        ----------
        txt : str
            Text.
        header : str
            Title of the modal.
        footer : str
            Extra info.

        Returns
        -------
            str
        """
        self.help = add_help(txt=txt, header=header, footer=footer)
        return self.help

    def write(self, outfile):
        """
        Writes an HTML output file.

        Parameters
        ----------
        outfile : str
            Name of the output file (include extension).
        """
        self.insert('{{toolbar}}', self.toolbar.write())
        if self.help is not None:
            self._replace('{{help}}', self.help)
        super(DiagramWriter, self).write(outfile)