"""
A widget that lets users set options from the root group of a model,
after the model has been instantiated but before setup has been called.

Timing is important here. After model has been instantiated, but before setup(),
the options dict should exist. User could still change option values at this point though.

A widget that lets them interact with common options. Setting any float/int. Checkboxes for bools.
If there is a set of specific values, provide them in a drop down.

Some options may need to be tagged as non-GUI-able. Ones that take function pointers, classes, or
instances seem like things you couldn't do in the gui. So we may need to add tags to the options
(recordable seems like a tag already), or just another metadata field perhaps.

Its critical the user can set these values before setup, because often some of these options change
the way the model is configured. After setup, these options should not be allowed to change.
"""

try:
    import ipywidgets as widgets
    from ipywidgets import DOMWidget, register
    from ipywidgets import interact, Layout
    from IPython.display import display
except Exception:
    widgets = None

from openmdao.utils.options_dictionary import OptionsDictionary
from openmdao.utils.general_utils import simple_warning


class OptionsWidget(object):
    """
    Widget to set options.

    Parameters
    ----------
    opts : OptionsDictionary
        options to edit.
    """

    def __init__(self, opts):
        """
        Initialize.
        """
        if widgets is None:
            simple_warning(f"ipywidgets is required to use {self.__class__.__name__}."
                           "To install it run `pip install openmdao[notebooks]`.")
            return

        _dict = opts._dict
        _widgets = []
        _style = {'description_width': 'initial', 'align-items': 'center'}

        messages = widgets.Output()

        @messages.capture(clear_output=True)
        def option_changed(change):
            owner = change['owner']
            newval = change['new']

            name = owner.description
            option = _dict[name]

            # if it's an arbitrary list, parse lines of text
            if option['types'] is list and option['values'] is None:
                newval = newval.strip().split('\n')

            try:
                option['val'] = newval
            except ValueError as err:
                print(str(err))

        for name, option in sorted(_dict.items()):
            val = option['val']
            types = option['types']
            values = option['values']
            desc = option['desc']

            if values:
                if types is list:
                    _widgets.append(widgets.SelectMultiple(
                        description=name,
                        tooltip=desc,
                        options=sorted(values),
                        value=val,
                        disabled=False,
                        style=_style
                    ))
                    continue
                else:
                    _widgets.append(widgets.Dropdown(
                        description=name,
                        tooltip=desc,
                        options=values,
                        value=val,
                        disabled=False,
                        style=_style
                    ))
                    continue

            upper = option['upper']
            lower = option['lower']

            if upper and lower:
                if isinstance(val, int):
                    _widgets.append(widgets.IntSlider(
                        description=name,
                        tooltip=desc,
                        min=lower,
                        max=upper,
                        value=val,
                        step=1,
                        disabled=False,
                        continuous_update=False,
                        orientation='horizontal',
                        readout=True,
                        readout_format='d',
                        style=_style
                    ))
                else:
                    _widgets.append(widgets.FloatSlider(
                        description=name,
                        tooltip=desc,
                        min=lower,
                        max=upper,
                        value=val,
                        disabled=False,
                        continuous_update=False,
                        orientation='horizontal',
                        readout=True,
                        readout_format='f',
                        style=_style
                    ))
                continue

            if isinstance(val, float):
                _widgets.append(widgets.FloatText(
                    description=name,
                    tooltip=desc,
                    min=lower,
                    max=upper,
                    value=val,
                    disabled=False,
                    continuous_update=False,
                    orientation='horizontal',
                    readout=True,
                    readout_format='f',
                    style=_style
                ))
                continue

            if isinstance(val, int):
                _widgets.append(widgets.IntText(
                    description=name,
                    tooltip=desc,
                    min=lower,
                    max=upper,
                    value=val,
                    step=1,
                    disabled=False,
                    continuous_update=False,
                    orientation='horizontal',
                    readout=True,
                    readout_format='d',
                    style=_style
                ))
                continue

            types = option['types']

            if types == list:
                _widgets.append(widgets.Textarea(
                    description=name,
                    tooltip=desc,
                    value='\n'.join(val),
                    continuous_update=False,
                    rows=5,
                    disabled=False,
                    style=_style
                ))
                continue

            # unhandled option type, just show value as uneditable text
            _widgets.append(widgets.Textarea(
                description=name,
                tooltip=desc,
                value=val,
                disabled=True,
                style=_style
            ))

        for wdgt in _widgets:
            wdgt.observe(option_changed, 'value')

        # sort widgets by how many rows they use
        _wdgt_rows = [(wdgt.rows if hasattr(wdgt, 'rows') else 1, wdgt) for wdgt in _widgets]
        _wdgt_rows.sort(key=lambda x: x[0])
        _widgets = [wdgt for _, wdgt in _wdgt_rows]

        box_layout = Layout(display='flex', flex_flow='row wrap')
        display(widgets.GridBox(children=_widgets, layout=box_layout))
        display(messages)
