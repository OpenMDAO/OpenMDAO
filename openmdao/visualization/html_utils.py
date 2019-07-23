"""
Functions to write HTML elements.
"""
import os

from six import iteritems, itervalues

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


def write_tags(tag, content='', attrs=None, cls=None, uid=None, new_lines=False, indent=0,
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
    cls : str or None
        The "class" attribute of the element.
    uid : str or None
        The "id" attribute of the element.
    new_lines : bool
        Make new line after tags.
    indent : int
        Indentation expressed in spaces.
        Defaults to 0.
    kwargs
        Alternative way to add element attributes. Use with attention, can overwrite some in-built
        python names as "class" or "id" if misused.
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
    if uid is not None:
        attrs['id'] = uid
    attrs = ' '.join(['{}="{}"'.format(k, v) for k, v in iteritems(attrs)])
    if isinstance(content, list):  # Convert iterable to string
        content = '\n'.join(content)
    return template.format(tag=tag, content=content, attributes=attrs, ls=line_sep, spaces=spaces)


def write_div(content='', attrs=None, cls=None, uid=None, indent=0, **kwargs):
    """
    Write an HTML div.

    Parameters
    ----------
    content : str or list(str)
        This goes into the body of the element.
    attrs : dict
        Attributes of the element.
    cls : str or None
        The "class" attribute of the element.
    uid : str or None
        The "id" attribute of the element.
    indent : int
        Indentation expressed in spaces.
    kwargs
        Alternative way to add element attributes. Use with attention, can overwrite some in-bult
        python names as "class" or "id" if misused.

    Returns
    -------
        str
    """
    return write_tags('div', content=content, attrs=attrs, cls=cls, uid=uid, new_lines=False,
                      indent=indent, **kwargs)


def write_style(content='', attrs=None, indent=0, **kwargs):
    """Write CSS."""
    default = {'type': "text/css"}
    if attrs is None:
        attrs = default
    else:
        attrs = default.update(attrs)
    return write_tags('style', content, attrs=attrs, new_lines=True, indent=indent, **kwargs)


def write_script(content='', attrs=None, indent=0, **kwargs):
    """Write JavaScript."""
    default = {'type': "text/javascript"}
    if attrs is None:
        attrs = default
    else:
        attrs = default.update(attrs)
    return write_tags('script', content, attrs=attrs, new_lines=True, indent=indent, **kwargs)


def write_paragraph(content):
    """Write a paragraph."""
    return write_tags(tag='p', content=content)


def read_files(filenames, directory, extension):
    """Read files (based on filenames) from a directory with a given extension."""
    libs = dict()
    for name in filenames:
        with open(os.path.join(directory, '.'.join([name, extension])), "r") as f:
            libs[name] = f.read()
    return libs

# Viewer API


def add_button(title, content='', uid=None, indent=0, **kwargs):
    """Add a button."""
    i = write_tags(tag='i', attrs={'class': content})
    attrs = {'title': title}
    return write_tags('button', cls="myButton", content=i, attrs=attrs, uid=uid,
                      new_lines=True, indent=indent, **kwargs)


def add_dropdown(title, id_naming=None, options=None, button_content='', header=None,
                 uid=None, indent=0, option_formatter=None, **kwargs):
    """Add a dropdown menu."""
    button = add_button(title=title, content=button_content)
    if header is None:
        header = title
    items = write_tags(tag='span', cls="fakeLink", content=header)

    if options is not None:
        for option in options:
            if option_formatter is not None:
                option = option_formatter(option)
            idx = "{}{}".format(id_naming, option)
            items += write_tags(tag='span', cls="fakeLink", uid=idx, content=option)

    attrs = {'class': 'dropdown-content'}
    menu = write_div(content=items, attrs=attrs, uid=uid)

    content = [button, menu]
    return write_div(content=content, cls='dropdown', indent=indent, **kwargs)


def add_help(txt, header='Instructions', footer=''):
    """
    Add a popup help.

    Parameters
    ----------
    txt : str
        Help message/instructions.
    header : str
        Message header.
    footer : str
        Additional info.

    Returns
    -------
        str
    """
    header_txt = write_tags(tag='span', cls='close', content='&times;', uid="idSpanModalClose")
    header_txt += header
    head = write_div(content=header_txt, cls="modal-header")
    foot = write_div(content=footer, cls="modal-footer")
    body = write_div(content=write_paragraph(txt), cls="modal-body")
    modal_content = write_div(content=[head, body, foot], cls="modal-content")
    return write_div(content=modal_content, cls="modal", uid="myModal")


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
    """Abstract class for user interface elements."""

    def __init__(self, indent=0):
        """Initialize."""
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
        kwargs : dict
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
        kwargs : dict
            Attributes passed to the dropdown element.

        Returns
        -------
            str
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
        return write_div(content=content, cls="button-group")


class Toolbar(UIElement):
    """
    A toolbar consists of button groups.

    Add button groups, and write it to get the HTML for the
    toolbar.
    """

    def add_button_group(self):
        """Add a group of buttons."""
        button_group = ButtonGroup(indent=self.indent + 4)
        self.items.append(button_group)
        return button_group

    def write(self):
        """
        Output the HTML code.

        Returns
        -------
            str
        """
        content = '\n\n'.join([item.write() for item in self.items])
        return write_div(content=content, cls="toolbar", uid="toolbarDiv")


class TemplateWriter(object):
    """
    Writes HTML files using templates.

    Opens an HTML template files, text can be inserted into the template, and writes a new HTML
    file with the replacements.
    """

    def __init__(self, filename, embeddable=False, title=None, styles=None):
        """Initialize."""
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
        """Insert text."""
        self._replace(ref=ref, txt=txt)

    def write(self, outfile):
        """Write the output file."""
        with open(outfile, 'w') as f:  # write output file
            f.write(self.template)


class DiagramWriter(TemplateWriter):
    """
    An HTML diagram writer.

    The diagram has a toolbar, which can be edited by adding
    button groups, dropdowns, buttons, etc. to this class.
    """

    def __init__(self, filename, embeddable=False, title=None, styles=None):
        """Initialize."""
        super(DiagramWriter, self).__init__(filename=filename, embeddable=embeddable, title=title,
                                            styles=styles)
        self.toolbar = Toolbar()
        self.help = None

    def add_help(self, txt, header='Instructions', footer=''):
        """
        Add a modal with instructions.

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
        Write an HTML output file.

        Parameters
        ----------
        outfile : str
            Name of the output file (include extension).
        """
        self.insert('{{toolbar}}', self.toolbar.write())
        if self.help is not None:
            self._replace('{{help}}', self.help)
        super(DiagramWriter, self).write(outfile)
