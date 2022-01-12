"""Widgets for accessing CaseReader in a Jupyter notebook."""

try:
    import ipywidgets as widgets
    from ipywidgets import interact, Layout
    from IPython.display import display
except Exception:
    widgets = None


try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

import numpy as np

from openmdao.recorders.case_reader import CaseReader
from openmdao.utils.general_utils import simple_warning


class CaseViewerWidget(object):
    """
    Widget to plot data from a CaseReader.

    Parameters
    ----------
    cr : CaseReader or str
        CaseReader or path to the recorded data file.
    source : str, optional
        Initial value for source.
    cases : 2-tuple of int
        Initial value for cases.
    x_axis : str, optional
        Initial value for x_axis.
    y_axis : str or list of str, optional
        Initial value for y_axis.
    """

    def __init__(self, cr, source=None, cases=None, x_axis=None, y_axis=None):
        """
        Initialize.
        """
        if widgets is None:
            simple_warning(f"ipywidgets is required to use {self.__class__.__name__}."
                           "To install it run `pip install openmdao[notebooks]`.")
            return

        if plt is None:
            simple_warning(f"matplotlib is required to use {self.__class__.__name__}."
                           "To install it run `pip install openmdao[visualization]`.")
            return

        if isinstance(cr, str):
            cr = CaseReader(cr)

        w_source = widgets.Dropdown(
            options=cr.list_sources(out_stream=None),
            description='Source:',
            disabled=False,
            layout=Layout(width='50%')
        )

        w_cases = widgets.IntRangeSlider(
            value=[0, 0],
            min=0,
            max=0,
            step=1,
            description='Cases',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d',
            layout=Layout(width='50%')
        )

        w_xaxis = widgets.Dropdown(
            options=['Iterations'],
            value='Iterations',
            description='X Axis:',
            disabled=False,
            layout=Layout(width='50%')
        )

        w_yaxis = widgets.SelectMultiple(
            options=cr.list_source_vars(w_source.value, out_stream=None)['outputs'],
            rows=len(cr.list_source_vars(w_source.value, out_stream=None)['outputs']),
            description='Y Axis',
            disabled=False,
            layout=Layout(width='50%')
        )

        def source_changed(*args):
            # keep selected vars
            oldvars = set(w_yaxis.value)

            # temporarily set harmless values
            w_cases.value = [0, 0]
            w_yaxis.value = []

            # update cases
            last_case = len(cr.list_cases(w_source.value, recurse=False, out_stream=None)) - 1
            w_cases.max = last_case
            w_cases.value = [0, last_case]

            # update yaxis
            yaxis = cr.list_source_vars(w_source.value, out_stream=None)['outputs']
            w_yaxis.options = yaxis
            w_yaxis.rows = min(len(yaxis), 15)
            w_yaxis.value = list(oldvars.intersection(set(yaxis)))

        w_source.observe(source_changed, 'value')
        source_changed()

        def cases_changed(*args):
            if w_cases.value[0] == w_cases.value[1]:
                w_xaxis.options = sorted(w_yaxis.options)
                # w_xaxis.value = w_yaxis.options[0]
            else:
                w_xaxis.options = ['Iterations']
                w_xaxis.value = 'Iterations'

        w_cases.observe(cases_changed, 'value')
        cases_changed()

        messages = widgets.Output()

        @messages.capture(clear_output=True)
        def plot_func(source, cases, xaxis, yaxis):
            # check if selected cases are not yet in sync with new source
            case_ids = cr.list_cases(source, recurse=False, out_stream=None)
            if cases[1] > len(case_ids) - 1:
                return

            # check if selected outputs are not yet in sync with new source
            case_outputs = cr.list_source_vars(source, out_stream=None)['outputs']
            if not (set(yaxis) <= set(case_outputs)):
                return

            case_nums = list(range(cases[0], cases[1] + 1))

            if xaxis == 'Iterations':
                x = case_nums
                selected_case_ids = [case_ids[n] for n in case_nums]
                for outvar in yaxis:
                    y = [cr.get_case(case_id).outputs[outvar] for case_id in selected_case_ids]
                    plt.plot(x, np.array(y), label=outvar)
                step = 1 if len(x) <= 15 else len(x) // 15
                plt.xticks(np.arange(cases[0], cases[1], step=step))
                plt.xlabel('Iterations')
            else:
                selected_case = cr.get_case(case_ids[case_nums[0]])
                x = selected_case.outputs[xaxis]
                if len(x.shape) > 1:
                    print("Output chosen for X axis must be one-dimensional, "
                          f"but {xaxis} has shape {x.shape}")
                    return
                for outvar in yaxis:
                    y = selected_case.outputs[outvar]
                    if y.shape[0] != x.shape[0]:
                        print(f"{xaxis} and {outvar} must have same first dimension, "
                              f"but have shapes {x.shape} and {y.shape}")
                        continue
                    plt.plot(x, np.array(y), label=outvar)
                plt.xticks(np.arange(x[0], x[-1]))
                plt.xlabel(xaxis)

            plt.grid(True)

            if yaxis:
                plt.legend(loc="best")

        plt.rcParams["figure.figsize"] = (16, 9)

        interact(plot_func, source=w_source, cases=w_cases, xaxis=w_xaxis, yaxis=w_yaxis,
                 disabled=False)

        display(messages)

        # set initial values
        if source:
            w_source.value = source
        if cases:
            w_cases.value = cases
        if x_axis:
            w_xaxis.value = x_axis
        if y_axis:
            if isinstance(y_axis, str):
                y_axis = [y_axis, ]
            w_yaxis.value = y_axis
