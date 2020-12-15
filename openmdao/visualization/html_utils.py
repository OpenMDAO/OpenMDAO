"""
Functions to write HTML elements.
"""
import os

_IND = 4  # indentation (spaces)


def head_and_body(head, body, attrs=None):
    """
    Make an html element from a head and body.

    Parameters
    ----------
    head : str
        Head of HTML.
    body : str
        Body of the HTML
    attrs : dict or None
        Attributes of the html element.
        Defaults to None.

    Returns
    -------
        str
    """
    # Wraps the head and body in tags
    doc_type = '<!doctype html>'
    head_elem = write_tags(tag='head', content=head, new_lines=True)
    body_elem = write_tags(tag='body', content=body, new_lines=True)
    content = '\n\n'.join([head_elem, body_elem])
    index = write_tags(tag='html', content=content, attrs=attrs, new_lines=True)
    return doc_type + '\n' + index


def write_tags(tag, content='', attrs=None, cls_attr=None, uid=None, new_lines=False, indent=0,
               **kwargs):
    """
    Write an HTML element enclosed in tags.

    Parameters
    ----------
    tag : str
        Name of the tag.
    content : str or list(str)
        This goes into the body of the element.
    attrs : dict or None
        Attributes of the element.
        Defaults to None.
    cls_attr : str or None
        The "class" attribute of the element.
    uid : str or None
        The "id" attribute of the element.
    new_lines : bool
        Make new line after tags.
    indent : int
        Indentation expressed in spaces.
        Defaults to 0.
    **kwargs : dict
        Alternative way to add element attributes. Use with attention, can overwrite some in-built
        python names as "class" or "id" if misused.

    Returns
    -------
    str
        HTML element enclosed in tags.
    """
    # Writes an HTML tag with element content and element attributes (given as a dictionary)
    line_sep = '\n' if new_lines else ''
    spaces = ' ' * indent
    template = '{spaces}<{tag} {attributes}>{ls}{content}{ls}</{tag}>\n'
    if attrs is None:
        attrs = {}
    attrs.update(kwargs)
    if cls_attr is not None:
        attrs['class'] = cls_attr
    if uid is not None:
        attrs['id'] = uid
    attrs = ' '.join(['{}="{}"'.format(k, v) for k, v in attrs.items()])
    if isinstance(content, list):  # Convert iterable to string
        content = '\n'.join(content)
    return template.format(tag=tag, content=content, attributes=attrs, ls=line_sep, spaces=spaces)


def write_div(content='', attrs=None, cls_attr=None, uid=None, indent=0, **kwargs):
    """
    Write an HTML div.

    Parameters
    ----------
    content : str or list(str)
        This goes into the body of the element.
    attrs : dict
        Attributes of the element.
    cls_attr : str or None
        The "class" attribute of the element.
    uid : str or None
        The "id" attribute of the element.
    indent : int
        Indentation expressed in spaces.
    **kwargs : dict
        Alternative way to add element attributes. Use with attention, can overwrite some in-bult
        python names as "class" or "id" if misused.

    Returns
    -------
        str
    """
    return write_tags('div', content=content, attrs=attrs, cls_attr=cls_attr, uid=uid,
                      new_lines=False, indent=indent, **kwargs)


def write_style(content='', attrs=None, indent=0, **kwargs):
    """
    Write CSS.

    Parameters
    ----------
    content : str or list(str)
        This goes into the body of the element.
    attrs : dict or None
        Attributes of the element.
        Defaults to None.
    indent : int
        Indentation expressed in spaces.
        Defaults to 0.
    **kwargs : dict
        Alternative way to add element attributes. Use with attention, can overwrite some in-built
        python names as "class" or "id" if misused.

    Returns
    -------
    str
        HTML for this CSS element.
    """
    default = {'type': "text/css"}
    if attrs is None:
        attrs = default
    else:
        attrs = default.update(attrs)
    return write_tags('style', content, attrs=attrs, new_lines=True, indent=indent, **kwargs)


def write_script(content='', attrs=None, indent=0, **kwargs):
    """
    Write JavaScript.

    Parameters
    ----------
    content : str or list(str)
        This goes into the body of the element.
    attrs : dict or None
        Attributes of the element.
        Defaults to None.
    indent : int
        Indentation expressed in spaces.
        Defaults to 0.
    **kwargs : dict
        Alternative way to add element attributes. Use with attention, can overwrite some in-built
        python names as "class" or "id" if misused.

    Returns
    -------
    str
        HTML for this JavaScript element.
    """
    default = {'type': "text/javascript"}
    if attrs is None:
        attrs = default
    else:
        attrs = default.update(attrs)
    return write_tags('script', content, attrs=attrs, new_lines=True, indent=indent, **kwargs)


def write_paragraph(content):
    """
    Write a paragraph.

    Parameters
    ----------
    content : str or list(str)
        This goes into the body of the element.

    Returns
    -------
    str
        HTML for this paragraph.
    """
    return write_tags(tag='p', content=content)


def read_files(filenames, directory, extension):
    """
    Read files (based on filenames) from a directory with a given extension.

    Parameters
    ----------
    filenames : list of str
        List of names of files to read.
    directory : str
        Pathname of directory.
    extension : str
        Extension of file names.

    Returns
    -------
    dict
        Dict of contents of files read with file names as keys.
    """
    libs = dict()
    for name in filenames:
        with open(os.path.join(directory, '.'.join([name, extension])), "r") as f:
            libs[name] = f.read()
    return libs

# Viewer API


def add_button(title, content='', uid=None, indent=0, **kwargs):
    """
    Add a button.

    Parameters
    ----------
    title : str
        Title of button.
    content : str or list(str)
        This goes into the body of the element.
    uid : str or None
        The "id" attribute of the element.
    indent : int
        Indentation expressed in spaces.
    **kwargs : dict
        Alternative way to add element attributes. Use with attention, can overwrite some in-bult
        python names as "class" or "id" if misused.

    Returns
    -------
    str
        HTML for this button.
    """
    i = write_tags(tag='i', attrs={'class': content})
    attrs = {'title': title}
    return write_tags('button', cls_attr="myButton", content=i, attrs=attrs, uid=uid,
                      new_lines=True, indent=indent, **kwargs)


def add_dropdown(title, id_naming=None, options=None, button_content='', header=None,
                 uid=None, indent=0, option_formatter=None, **kwargs):
    """
    Add a dropdown menu.

    Parameters
    ----------
    title : str
        Title of button.
    id_naming : str
        Prefix for the id's for the items in the menu.
    options : list
        List of options for the menu.
    button_content : str
        Class of icon fonts used for the menu buttons.
    header : str
        Top item in the menu. It describes the purpose of the menu.
    uid : str or None
        The "id" attribute of the element.
    indent : int
        Indentation expressed in spaces.
    option_formatter : func
        Function used to format the displayed menu options using the values from options arg.
    **kwargs : dict
        Alternative way to add element attributes. Use with attention, can overwrite some in-bult
        python names as "class" or "id" if misused.

    Returns
    -------
    str
        HTML for this dropdown.
    """
    button = add_button(title=title, content=button_content)
    if header is None:
        header = title
    items = write_tags(tag='span', cls_attr="fakeLink", content=header)

    if options is not None:
        for option in options:
            if option_formatter is not None:
                option = option_formatter(option)
            idx = "{}{}".format(id_naming, option)
            items += write_tags(tag='span', cls_attr="fakeLink", uid=idx, content=option)

    attrs = {'class': 'dropdown-content'}
    menu = write_div(content=items, attrs=attrs, uid=uid)

    content = [button, menu]
    return write_div(content=content, cls_attr='dropdown', indent=indent, **kwargs)


def add_help(txt, diagram_filepath, header='Instructions', footer=''):
    """
    Add a popup help.

    Parameters
    ----------
    txt : str
        Help message/instructions.
    diagram_filepath : str
        File path to the diagram file in SVG format.
    header : str
        Message header.
    footer : str
        Additional info.

    Returns
    -------
        str
    """
    header_txt = write_tags(tag='span', cls_attr='close', content='&times;', uid="idSpanModalClose")
    header_txt += header
    head = write_div(content=header_txt, cls_attr="modal-header")
    foot = write_div(content=footer, cls_attr="modal-footer")
    body = write_div(content=write_paragraph(txt), cls_attr="modal-body")
    toolbar_help_header = write_div(content='Toolbar Help', cls_attr="modal-section-header")

    with open(diagram_filepath, "r") as f:
        help_diagram = f.read()
    help_diagram = write_div(content=help_diagram, cls_attr="toolbar-help")

    modal_content = write_div(content=[head, body, toolbar_help_header, help_diagram, foot],
                              cls_attr="modal-content")
    return write_div(content=modal_content, cls_attr="modal", uid="myModal")


def add_title(txt, heading='h1', align='center'):
    """
    Add a title heading.

    Parameters
    ----------
    txt : str
        Title text.
    heading : str
        Heading. Options are "h1" to "h6".
        Defaults to "h1"
    align : str
        Defaults to "center"

    Returns
    -------
        str
    """
    title = write_tags(tag=heading, content=txt)
    style = "text-align: {}".format(align)
    return write_div(content=title, uid="maintitle", attrs={'style': style})


class UIElement(object):
    """
    Abstract class for user interface elements.

    Attributes
    ----------
    items: list
        List of UI elements contained by this element.
    indent : int
        Number of spaces for indent.
    """

    def __init__(self, indent=0):
        """
        Initialize.

        Parameters
        ----------
        indent : int
            Number of spaces to indent.
        """
        self.items = []
        self.indent = indent


class ButtonGroup(UIElement):
    """
    Button group, which consists of buttons and dropdowns.

    Write it to get the HTML for the button group.
    """

    def add_button(self, title, content='', uid=None, **kwargs):
        """
        Add a button to the button group.

        Parameters
        ----------
        title : str
            Name to be shown.
        content : str, optional
            The content of the element.
        uid : str or None
            ID.
        **kwargs : dict
            Attributes passed to the button element.

        Returns
        -------
            str
        """
        button = add_button(title, content=content, uid=uid, indent=self.indent + _IND,
                            **kwargs)
        self.items.append(button)
        return button

    def add_dropdown(self, title, id_naming=None, options=None, button_content='', header=None,
                     uid=None, option_formatter=None, **kwargs):
        """
        Add a dropdown to the button group.

        Parameters
        ----------
        title : str
            Name to be shown.
        id_naming : str or None, optional
            ID of an item will be id_naming + option
            Defaults to None.
        options : list(str) or None, optional
            Items of the dropdown. Can be None, if filled with a script.
            Defaults to None.
        button_content : str, optional
            Content of the button.
        header : str or None, optional
            First item in the dropdown. Defaults to the title.
        uid : str or None, optional
            ID.
        option_formatter : None or callable, optional
            Text formatter for the dropdown items. Called with one item.
            Defaults to None.
        **kwargs : dict
            Attributes passed to the dropdown element.

        Returns
        -------
        str
            HTML for dropdown.
        """
        dropdown = add_dropdown(title=title, id_naming=id_naming, options=options,
                                button_content=button_content, header=header,
                                uid=uid, indent=self.indent + _IND,
                                option_formatter=option_formatter, **kwargs)
        self.items.append(dropdown)
        return dropdown

    def write(self):
        """
        Output the HTML code.

        Returns
        -------
            str
        """
        content = '\n\n'.join(self.items)
        return write_div(content=content, cls_attr="button-group")


class Toolbar(UIElement):
    """
    A toolbar consists of button groups.

    Add button groups, and write it to get the HTML for the
    toolbar.
    """

    def add_button_group(self):
        """
        Add a group of buttons.

        Returns
        -------
            ButtonGroup
        """
        button_group = ButtonGroup(indent=self.indent + 4)
        self.items.append(button_group)
        return button_group

    def write(self):
        """
        Output the HTML code.

        Returns
        -------
        str
            HTML div enclosed in tags.
        """
        content = '\n\n'.join([item.write() for item in self.items])
        return write_div(content=content, cls_attr="toolbar", uid="toolbarDiv")


class TemplateWriter(object):
    """
    Writes HTML files using templates.

    Opens an HTML template file, text can be inserted into the template, and writes a new HTML
    file with the replacements.

    Attributes
    ----------
    template : str
        Contents of template file.
    """

    def __init__(self, filename, embeddable=False, title=None, styles=None):
        """
        Initialize.

        Parameters
        ----------
        filename : str
            Name of template file.
        embeddable : bool
            If true, create file so that it can be embedded in a webpage.
        title : str
            Title of diagram.
        styles : dict
            Dictionary of CSS styles.
        """
        # Load template
        with open(filename, "r") as f:
            self.template = template = f.read()

        if styles is not None:
            style_elems = '\n\n'.join([write_style(content=s) for s in styles.values()])

            if embeddable:
                self.template = '\n\n'.join([style_elems, template])
            else:
                meta = '<meta http-equiv="Content-Type" content="text/html; charset=UTF-8">'
                if title:
                    title_tag = "<title>%s</title>" % title
                    head = '\n\n'.join([title_tag, meta, style_elems])  # Write styles to head
                else:
                    head = '\n\n'.join([meta, style_elems])  # Write styles to head
                self.template = head_and_body(head=head, body=template)

        if title is not None:
            self._replace('{{title}}', add_title(title))

    def _replace(self, ref, txt):
        # Replace a reference in the template file with a text
        self.template = self.template.replace(ref, txt)

    def insert(self, ref, txt):
        """
        Insert text.

        Parameters
        ----------
        ref : str
            Reference string in template file.
        txt : str
            String used to replace text in template file.
        """
        self._replace(ref=ref, txt=txt)

    def write(self, outfile):
        """
        Write the output file.

        Parameters
        ----------
        outfile : str
            Path name for file to write to.
        """
        with open(outfile, 'w') as f:  # write output file
            f.write(self.template)


class DiagramWriter(TemplateWriter):
    """
    An HTML diagram writer.

    The diagram has a toolbar, which can be edited by adding
    button groups, dropdowns, buttons, etc. to this class.

    Attributes
    ----------
    toolbar : Toolbar
        The toolbar with button groups for this diagram.
    help : Toolbar
        String of HTML for the help dialog.
    """

    def __init__(self, filename, embeddable=False, title=None, styles=None):
        """
        Initialize.

        Parameters
        ----------
        filename : str
            Name of template file.
        embeddable : bool
            If true, create file so that it can be embedded in a webpage.
        title : str
            Title of diagram.
        styles : dict
            Dictionary of CSS styles.
        """
        super().__init__(filename=filename, embeddable=embeddable, title=title, styles=styles)
        self.toolbar = Toolbar()
        self.help = None

    def add_help(self, txt, diagram_filepath, header='Instructions', footer=''):
        """
        Add a modal with instructions.

        Parameters
        ----------
        txt : str
            Text.
        diagram_filepath : str
            File path to the diagram file in SVG format.
        header : str
            Title of the modal.
        footer : str
            Extra info.

        Returns
        -------
        str
            String of HTML for the help dialog.
        """
        self.help = add_help(txt=txt, diagram_filepath=diagram_filepath, header=header,
                             footer=footer)
        return self.help

    def write(self, outfile):
        """
        Write an HTML output file.

        Parameters
        ----------
        outfile : str
            Name of the output file (include extension).
        """
        self.insert('{{toolbar}}', self.toolbar.write())
        if self.help is not None:
            self._replace('{{help}}', self.help)
        super().write(outfile)
