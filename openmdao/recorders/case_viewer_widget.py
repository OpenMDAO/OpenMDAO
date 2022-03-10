"""Widgets for accessing CaseReader in a Jupyter notebook."""

import ast
import pathlib
import re
import webbrowser

import openmdao.api as om
import numpy as np

_DEBUG = False

try:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib import cm
except:
    mpl = None

try:
    import ipywidgets as ipw
except ImportError:
    ipw = None

try:
    from IPython.display import display
    from IPython import get_ipython
except ImportError:
    get_ipython = None


_func_map = {'None': lambda x: x,
            'ravel': np.ravel,
            'min': np.min,
            'minabs': lambda x: np.min(np.abs(x)),
            'max': np.max,
            'maxabs': lambda x: np.max(np.abs(x)),
            'norm': np.linalg.norm}


def _apply_transform(data, transform):
    """
    Apply a function from _func_map to the data based on the specified transform.

    Parameters
    ----------
    data : np.array
        The data to be transformed.
    transform : str
        The name of the transform to be applied.  This will be looed up in _func_map.
    """
    _func_map[transform]
    if _func_map[transform]:
        out = _func_map[transform](data)
        return np.atleast_2d(np.asarray(out))
    return data


def _apply_slice(data, s):
    """
    Apply slice s (given as a string) to the data.

    Parameters
    ----------
    data : np.array
        The data to be sliced.
    s : str
        A string representation of the slice, e.g. '[...]', '[0, :]', etc.

    Returns
    -------
    np.array
        The sliced data.
    """
    safe_funcs = {'index_exp': np.index_exp, '__builtins__': None}
    sl = eval(f'index_exp{s}', safe_funcs, None)
    return data[sl]


def _get_output_meta(cr, case_name, var):
    """
    Return the metadata for the variable of the given name in the output cases.

    Parameters
    ----------
    cr : om.CaseReader
        The case reader housing the data.
    case_name : str
        The case from which the outputs with avaialble metadata is to be returned.
    var : str
        The output whose metadata is desired.

    Returns
    -------
    list
        A dictionary of the metadata for the given output
    """
    output_vars = set()
    case = cr.get_case(case_name)
    case_outputs = case.list_outputs(prom_name=True, units=True, shape=True, val=False, residuals=False,
                                     out_stream=None)

    for _, meta in case_outputs:
        if meta['prom_name'] == var:
            return meta
    else:
        raise KeyError(f'No output named {var} found')


def _get_output_vars(cr, case_names):
    """
    Return a set of variable names whose values are present in at lease one of the given cases.

    Parameters
    ----------
    cr : om.CaseReader
        The CaseReader housing the data.
    case_names : Iterable of str
        The case_names from which the outputs with avaialble residuals out are to be returned.

    Returns
    -------
    list
        A list of the variables with avaialble residuals in at least one of the given cases.
    """
    output_vars = set()
    for case_name in case_names:
        case = cr.get_case(case_name)
        case_outputs = case.list_outputs(prom_name=True, units=False, shape=False, val=True, residuals=False,
                                         out_stream=None)
        output_vars |= {meta['prom_name'] for abs_path, meta in case_outputs if isinstance(meta['val'], np.ndarray)}
    return sorted(list(output_vars))


def _get_resids_vars(cr, case_names):
    """
    Return a set of variable names whose residuals are present in at lease one of the given cases.

    Parameters
    ----------
    cr : om.CaseReader
        The CaseReader housing the data.
    case_names : Iterable of str
        The case_names from which the outputs with avaialble residuals out are to be returned.

    Returns
    -------
    list
        A list of the variables with avaialble residuals in at least one of the given cases.
    """
    resid_vars = set()
    for case_name in case_names:
        case = cr.get_case(case_name)
        case_outputs = case.list_outputs(prom_name=True, units=True, shape=True, val=False, residuals=True,
                                         out_stream=None)
        resid_vars |= {meta['prom_name'] for abs_path, meta in case_outputs if isinstance(meta['resids'], np.ndarray)}
    return sorted(list(resid_vars))


def _get_opt_vars(cr, case_names, var_type=None):
    """
    Return a set of variable names whose outputs are present in at lease one of the given cases and are part
    of the optimization variables in at least one of the cases.

    Parameters
    ----------
    cr : om.CaseReader
        The CaseReader housing the data.
    case_names : Iterable of str
        The case_names from which the outputs with avaialble residuals out are to be returned.
    var_type : None or str
        One of 'desvars', 'constraints', 'objectives', or None.

    Returns
    -------
    list
        A list of the variables with avaialble residuals in at least one of the given cases.
    """
    vars = set()
    for case_name in case_names:
        case = cr.get_case(case_name)
        if var_type in ('desvars', None):
            vars |= case.get_design_vars().keys()
        if var_type in ('constraints', None):
            vars |= case.get_constraints().keys()
        if var_type in ('objectives', None):
            vars |= case.get_objectives().keys()
    return sorted(list(vars))


def _get_resids_val(case, prom_name):
    """
    Retrieve residuals associated with the given output in the given case.

    Parameters
    ----------
    case : om.Case
        The CaseReader Case from which residuals are to be retrieved.
    prom_name : str
        The promoted name of the output whose residuals are to be retrieved.

    Returns
    -------
    np.ndarray
        The residuals of the given output in the given case.

    """
    listed_op = case.list_outputs(prom_name=True, residuals=True, val=False, out_stream=None)
    d = {meta['prom_name']: meta for _, meta in listed_op}
    return d[prom_name]['resids']


class CaseViewer(object):
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

    Attributes
    ----------

    """
    def __init__(self, f, source=None, cases=None, x_axis=None, y_axis=None):
        """
        Initialize the case viewer interface.
        """
        if mpl is None:
            raise RuntimeError('CaseViewer requires matplotlib and ipympl')
        if get_ipython is None:
            raise RuntimeError('CaseViewer requires jupyter')
        if ipw is None:
            raise RuntimeError('CaseViewer requires ipywidgets')
        if get_ipython() is None:
            raise RuntimeError('CaseViewer must be run from within a Jupyter notebook.')

        get_ipython().run_line_magic('matplotlib', 'widget')

        self._outputs = {}
        self._desvars = set()
        self._constraints = set()
        self._objectives = set()
        self._resids = set()

        self._case_reader = om.CaseReader(f) if isinstance(f, str) else f

        self._cmap = cm.viridis

        self._case_index_str = 'Case Index'

        self._filename = self._case_reader._filename

        self._make_gui()

        self._register_callbacks()

        self._fig, self._ax = plt.subplots(1, 1, figsize=(9, 9/1.6), tight_layout=True)
        # gs = mpl.gridspec.GridSpec(1, 20, figure=self._fig)
        # self._ax = self._fig.add_subplot(gs[0, :-1])
        # self._cbar_ax = self._fig.add_subplot(gs[0, -1])
        # self._cbar_ax.set_axis_off()
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        self._scalar_mappable = cm.ScalarMappable(norm=norm, cmap=self._cmap)
        self._colorbar = self._fig.colorbar(self._scalar_mappable, label='Case Index')

        self._update_source_options()
        self._update_case_select_options()
        self._update_var_select_options('x')
        self._update_var_select_options('y')
        self._update_var_info('x')
        self._update_var_info('y')

    def _make_gui(self):
        """
        Define the widgets for the CaseViewer and displays them.
        """
        self._widgets = {}

        self._widgets['source_select'] = ipw.Dropdown(description='Source',
                                                      layout=ipw.Layout(width='30%', height='auto'))

        self._widgets['cases_select'] = ipw.SelectMultiple(description='Cases',
                                                          layout=ipw.Layout(width='40%', height='auto'))

        self._widgets['case_select_button'] = ipw.Button(description='Select ' + '\u27F6',
                                                         layout={'width': '100%'})

        self._widgets['case_select_all_button'] = ipw.Button(description='Select All ' + '\u27F9',
                                                             layout={'width': '100%'})

        self._widgets['case_remove_button'] = ipw.Button(description='Remove ' + '\u274c',
                                                         layout={'width': '100%'})

        self._widgets['cases_list'] = ipw.Select(layout=ipw.Layout(width='40%', height='auto'))

        self._widgets['x_filter'] = ipw.Text('', description='X-Axis Filter',
                                             layout=ipw.Layout(width='49%', height='auto'))

        var_types_list = ['outputs', 'optimization', 'desvars', 'constraints', 'objectives', 'residuals']


        self._widgets['x_var_type'] = ipw.Dropdown(options=var_types_list,
                                                   description='X Var Type',
                                                   value='outputs',
                                                   layout={'width': '49%'})

        self._widgets['x_select'] = ipw.Select(description='X-Axis',
                                               layout=ipw.Layout(width='auto', height='auto'))


        self._widgets['x_transform_select'] = ipw.Dropdown(options=_func_map.keys(),
                                                           value='None',
                                                           description='X Transform',
                                                           layout=ipw.Layout(width='95%', height='auto'))

        self._widgets['x_slice'] = ipw.Text('[...]', description='X Slice',
                                            layout=ipw.Layout(width='95%', height='auto'))

        self._widgets['x_info'] = ipw.HTML(value='', description='X Shape', layout={'width': '95%'})

        self._widgets['x_scale'] = ipw.Dropdown(options=['linear', 'log'], value='linear', description='X Scale',
                                                layout={'width': '95%'})

        self._widgets['y_filter'] = ipw.Text('', description='Y-Axis Filter',
                                             layout={'width': '49%', 'height': 'auto'})

        self._widgets['y_var_type'] = ipw.Dropdown(options=var_types_list,
                                                   description='Y Var Type',
                                                   value='outputs',
                                                   layout={'width': '49%'})

        self._widgets['y_select'] = ipw.Select(description='Y-Axis',
                                               layout=ipw.Layout(width='auto', height='auto'))

        self._widgets['y_transform_select'] = ipw.Dropdown(options=_func_map.keys(),
                                                           value='None',
                                                           description='Y Transform',
                                                           layout=ipw.Layout(width='95%', height='auto'))

        self._widgets['y_slice'] = ipw.Text('[...]',
                                description='Y Slice',
                                layout=ipw.Layout(width='95%', height='auto'))

        self._widgets['y_info'] = ipw.HTML(value='', description='Y Shape', layout={'width': '95%'})

        self._widgets['y_scale'] = ipw.Dropdown(options=['linear', 'log'], value='linear', description='Y Scale',
                                                layout={'width': '95%'})

        self._widgets['case_slider'] = ipw.IntSlider(value=1, min=0, max=1, step=1, description='Case #',
                                                     disabled=False, continuous_update=True, orientation='horizontal',
                                                     readout=True, readout_format='d', layout={'width': '95%'})

        self._widgets['debug_output'] = ipw.Output(description='',
                                                   layout={'border': '0px solid black', 'width': '95%',
                                                           'height': '400'})

        display(ipw.VBox([self._widgets['source_select'],
                          ipw.HBox([self._widgets['cases_select'], ipw.VBox([self._widgets['case_select_button'],
                                                                             self._widgets['case_select_all_button'],
                                                                             self._widgets['case_remove_button']]),
                                    self._widgets['cases_list']], layout={'width': '95%'}),
                          ipw.HBox([ipw.VBox([ipw.HBox([self._widgets['x_filter'], self._widgets['x_var_type']]),
                                              self._widgets['x_select']],
                                             layout={'width': '50%'}),
                                    ipw.VBox([self._widgets['x_info'],
                                              self._widgets['x_slice'],
                                              self._widgets['x_transform_select'],
                                              self._widgets['x_scale']],
                                             layout={'width': '20%'}),
                                    ]),
                          ipw.HBox([ipw.VBox([ipw.HBox([self._widgets['y_filter'], self._widgets['y_var_type']]),
                                              self._widgets['y_select']],
                                             layout={'width': '50%'}),
                                    ipw.VBox([self._widgets['y_info'],
                                              self._widgets['y_slice'],
                                              self._widgets['y_transform_select'],
                                              self._widgets['y_scale']],
                                             layout={'width': '20%'})
                                    ]),
                          self._widgets['case_slider'],
                          self._widgets['debug_output']]))

    def _update_source_options(self):
        """
        Update the contents of the source selection dropdown menu.
        """
        sources = self._case_reader.list_sources(out_stream=None)
        self._widgets['source_select'].options = sources
        self._widgets['source_select'].value = sources[0]

    def _update_var_info(self, axis):
        """
        Update the variable info displayed.

        Parameters
        ----------
        axis : str
            'x' or 'y' - case insensitive.
        """
        if axis.lower() not in ('x', 'y'):
            raise ValueError(f'Unknown axis: {axis}')

        src = self._widgets['source_select'].value
        cases = self._widgets['cases_list'].options

        if not cases:
            self._widgets[f'{axis}_info'].value = 'N/A'
            return

        var = self._widgets[f'{axis}_select'].value

        if var == self._case_index_str:
            shape = (len(cases),)
        elif var is None:
            shape = 'N/A'
        else:
            meta = _get_output_meta(self._case_reader, cases[0], var)
            shape = meta['shape']

        self._widgets[f'{axis}_info'].value = f'{shape}'

    def _update_case_select_options(self):
        """
        Update the available cases listed in the source_select widget.
        """
        src = self._widgets['source_select'].value
        avialable_cases = self._case_reader.list_cases(source=src, recurse=False, out_stream=None)
        self._widgets['cases_select'].options = avialable_cases
        self._update_case_slider()

    def _update_var_select_options(self, axis):
        """
        Update the variables available for plotting.

        Parameters
        ----------
        axis : str
            'x' or 'y' - case insensitive.
        """
        with self._widgets['debug_output']:
            if axis.lower() not in ('x', 'y'):
                raise ValueError(f'Unknown axis: {axis}')

            src = self._widgets['source_select'].value
            cases = self._widgets['cases_list'].options

            if not cases:
                self._widgets[f'{axis}_select'].options = []
                self._widgets[f'{axis}_info'].value = 'N/A'
                return

            w_var_select = self._widgets[f'{axis}_select']
            var_filter = self._widgets[f'{axis}_filter'].value
            var_select = w_var_select.value
            var_type = self._widgets[f'{axis}_var_type'].value

            if var_type == 'optimization':
                vars = _get_opt_vars(self._case_reader, cases)
            elif var_type in ('desvars', 'constraints', 'objectives'):
                vars = _get_opt_vars(self._case_reader, cases, var_type=var_type)
            elif var_type == 'residuals':
                vars = _get_resids_vars(self._case_reader, cases)
            else:
                vars = _get_output_vars(self._case_reader, cases)

            # We have a list of available vars, now filter it.
            r = re.compile(var_filter)
            filtered_list = list(filter(r.search, vars))

            w_var_select.options = [self._case_index_str] + filtered_list if axis == 'x' else filtered_list

            self._update_var_info(axis)

    def _update_case_slider(self):
        """
        Update the extents of the case slider.
        """
        n = len(self._widgets['cases_list'].options)
        self._widgets['case_slider'].max = n
        self._widgets['case_slider'].value = n

    def _register_callbacks(self):
        """
        Register callback functions with the widgets.
        """
        self._widgets['debug_output'].clear_output()
        with self._widgets['debug_output']:

            self._widgets['source_select'].observe(self._callback_select_source)
            self._widgets['case_select_button'].on_click(self._callback_select_case)
            self._widgets['case_select_all_button'].on_click(self._callback_select_all_cases)
            self._widgets['case_remove_button'].on_click(self._callback_remove_case)

            self._widgets['cases_list'].observe(self._callback_case_list_select)

            self._widgets['x_filter'].observe(self._callback_filter_vars, 'value')
            self._widgets['y_filter'].observe(self._callback_filter_vars, 'value')

            self._widgets['x_var_type'].observe(self._callback_filter_vars, 'value')
            self._widgets['y_var_type'].observe(self._callback_filter_vars, 'value')

            self._widgets['x_select'].observe(self._callback_select_var, 'value')
            self._widgets['y_select'].observe(self._callback_select_var, 'value')

            self._widgets['x_slice'].observe(self._callback_change_slice, 'value')
            self._widgets['y_slice'].observe(self._callback_change_slice, 'value')

            self._widgets['x_scale'].observe(self._callback_change_scale, 'value')
            self._widgets['y_scale'].observe(self._callback_change_scale, 'value')

            self._widgets['x_transform_select'].observe(self._callback_select_transform, 'value')
            self._widgets['y_transform_select'].observe(self._callback_select_transform, 'value')

            self._widgets['case_slider'].observe(self._callback_case_slider, 'value')

    def _callback_select_source(self, *args):
        """
        Repopulate cases_select with cases from the chosen source.

        Parameters
        ----------
        args : tuple
            The information passed by the widget upon callback.
        """
        self._widgets['debug_output'].clear_output()
        with self._widgets['debug_output']:
            self._update_case_select_options()

    def _callback_select_case(self, *args):
        """
        Add the selected case(s) to the chosen cases list.

        Parameters
        ----------
        args : tuple
            The information passed by the widget upon callback.
        """
        self._widgets['debug_output'].clear_output()
        with self._widgets['debug_output']:
            clw = self._widgets['cases_list']
            current = clw.options
            new = self._widgets['cases_select'].value
            numeric_sorter = lambda case_name: (case_name.split('|')[0], int(case_name.split('|')[-1]))
            self._widgets['cases_list'].options = sorted(list(set(current + new)), key=numeric_sorter)
            self._update_case_slider()
            self._update_var_select_options('x')
            self._update_var_select_options('y')
            self._update_plot()

    def _callback_case_list_select(self, *args):
        """
        Update the plot when a different case is selected in the cases list.

        Parameters
        ----------
        args : tuple
            The information passed by the widget upon callback.
        """
        self._widgets['debug_output'].clear_output()
        with self._widgets['debug_output']:
            self._update_plot()

    def _callback_case_slider(self, *args):
        """
        Update the plot when the case slider is changed.

        Parameters
        ----------
        args : tuple
            The information passed by the widget upon callback.
        """
        self._widgets['debug_output'].clear_output()
        with self._widgets['debug_output']:
            self._update_plot()

    def _callback_select_all_cases(self, *args):
        """
        Add all available cases to the cases list.

        Parameters
        ----------
        args : tuple
            The information passed by the widget upon callback.
        """
        self._widgets['debug_output'].clear_output()
        with self._widgets['debug_output']:
            clw = self._widgets['cases_list']
            current = clw.options
            new = self._widgets['cases_select'].options
            numeric_sorter = lambda case_name: (case_name.split('|')[0], int(case_name.split('|')[-1]))
            self._widgets['cases_list'].options = sorted(list(set(current + new)), key=numeric_sorter)
            self._update_case_slider()
            self._update_var_select_options('x')
            self._update_var_select_options('y')
            self._update_plot()

    def _callback_remove_case(self, *args):
        """
        Remove the selected case from the chosen cases list widget.

        Parameters
        ----------
        args : tuple
            The information passed by the widget upon callback.
        """
        self._widgets['debug_output'].clear_output()
        with self._widgets['debug_output']:
            clw = self._widgets['cases_list']
            new_list = list(clw.options)
            if clw.value in new_list:
                new_list.remove(clw.value)
            clw.options = new_list
            self._update_var_select_options('x')
            self._update_var_select_options('y')
            self._update_plot()

    def _callback_filter_vars(self, *args):
        """
        Update the plot and the available variables when the filtering criteria is changed.

        Parameters
        ----------
        args : tuple
            The information passed by the widget upon callback.
        """
        event = args[0]
        self._widgets['debug_output'].clear_output()
        with self._widgets['debug_output']:
            w = event['owner']
            axis = 'x' if w is self._widgets['x_filter'] or w is self._widgets['x_var_type'] else 'y'
            self._update_var_select_options(axis)
            self._update_plot()

    def _callback_select_var(self, *args):
        """
        Update the variable info and the plot when a new variable is chosen.

        Parameters
        ----------
        args : tuple
            The information passed by the widget upon callback.
        """
        event = args[0]
        self._widgets['debug_output'].clear_output()
        with self._widgets['debug_output']:
            w = event['owner']
            s = w.value
            src = self._widgets['source_select'].value
            cases = self._widgets['cases_list'].options
            axis = 'x' if w is self._widgets['x_select'] else 'y'

            self._update_var_info(axis)

            if s is None:
                self._ax.clear()
            else:
                self._update_plot()

    def _callback_change_slice(self, *args):
        """
        Update the plot when a new, valid slice is provided.

        Parameters
        ----------
        args : tuple
            The information passed by the widget upon callback.
        """
        event = args[0]
        self._widgets['debug_output'].clear_output()
        with self._widgets['debug_output']:
            w = event['owner']
            s = w.value

            if s.startswith('[') and s.endswith(']'):
                self._update_plot()

    def _callback_change_scale(self, *args):
        """
        Update the plot when the x or y axis scale is changed.

        Parameters
        ----------
        args : tuple
            The information passed by the widget upon callback.
        """
        self._widgets['debug_output'].clear_output()
        with self._widgets['debug_output']:
            event = args[0]
            w = event['owner']
            if w is self._widgets['x_scale']:
                self._ax.set_xscale(w.value)
            else:
                self._ax.set_yscale(w.value)

    def _callback_select_transform(self, *args):
        """
        Update the plot when a new transformation is choen for the x or y variable.

        Parameters
        ----------
        args : tuple
            The information passed by the widget upon callback.
        """
        self._update_plot()

    def _redraw_plot(self):
        """
        Update the plot area by plotting one variable vs another over one or more cases.
        """
        x_min = y_min = 1E16
        x_max = y_max = -1E16

        cases = self._widgets['cases_list'].options

        x_slice = self._widgets['x_slice'].value
        y_slice = self._widgets['y_slice'].value

        x_transform = self._widgets['x_transform_select'].value
        y_transform = self._widgets['y_transform_select'].value

        x_var = self._widgets['x_select'].value
        y_var = self._widgets['y_select'].value

        x_var_type = self._widgets['x_var_type'].value
        y_var_type = self._widgets['y_var_type'].value

        selected_case_idx = self._widgets['case_slider'].value

        max_size = 0

        self._widgets['debug_output'].clear_output()
        with self._widgets['debug_output']:
            for i, case_name in enumerate(cases):
                alpha = 1.0 if (selected_case_idx >= len(cases) or i == selected_case_idx) else 0.1
                lw = 2.0 if (selected_case_idx >= len(cases) or i == selected_case_idx) else 0.05
                case = self._case_reader.get_case(case_name)

                if y_var_type == 'residuals':
                    y_val = _get_resids_val(case, y_var)
                else:
                    y_val = case.get_val(y_var)

                try:
                    y_val = _apply_slice(y_val, y_slice)
                except:
                    if _DEBUG:
                        print(f'Error while applying Y slice: {y_slice}')
                    continue
                y_val = _apply_transform(y_val, y_transform)

                if x_var != self._case_index_str:
                    if x_var_type == 'residuals':
                        x_val = _get_resids_val(case, x_var)
                    else:
                        x_val = self._case_reader.get_case(case_name).get_val(x_var)
                else:
                    x_val = i * np.ones_like(y_val)

                try:
                    x_val = _apply_slice(x_val, x_slice)
                except:
                    if _DEBUG:
                        print(f'Error while applying X slice: {x_slice}')
                    continue
                x_val = _apply_transform(x_val, x_transform)

                if x_val is None or y_val is None:
                    continue

                if x_val.shape[0] != y_val.shape[0]:
                    print(f'Incompatible shapes: x.shape = {x_val.shape}  y.shape = {y_val.shape}.')
                    print('Size along first axis must agree.')
                    return

                max_size = max(max_size, x_val.size)

                x_min = min(x_min, np.min(x_val))
                x_max = max(x_max, np.max(x_val))
                y_min = min(y_min, np.min(y_val))
                y_max = max(y_max, np.max(y_val))

                if x_var == self._case_index_str:
                    self._ax.scatter(x_val, y_val, c=np.arange(x_val.size), s=20, cmap=self._cmap, alpha=alpha)
                else:
                    self._ax.plot(x_val, y_val,
                                  color=self._cmap(float(i)/len(cases)),
                                  marker='o',
                                  linestyle='-',
                                  linewidth=lw,
                                  markersize=2,
                                  alpha=alpha)

                self._fig.canvas.flush_events()

            x_margin = (x_max - x_min) * 0.05
            x_margin = 0.1 if x_margin < 1.0E-16 else x_margin
            y_margin = (y_max - y_min) * 0.05
            y_margin = 0.1 if y_margin < 1.0E-16 else y_margin

            bad_x_bounds = np.any(x.isinf() or x.isnan() for x in [x_min, x_max, x_margin])
            bad_y_bounds = np.any(x.isinf() or x.isnan() for x in [x_min, x_max, x_margin])

            if not bad_x_bounds:
                self._ax.set_xlim(x_min - x_margin, x_max + x_margin)
            if not bad_y_bounds:
                self._ax.set_ylim(y_min - y_margin, y_max + y_margin)

            # Add the colorbar.  Color shows the index of each point in its vector if the x-axis is Case Index,
            # otherwise it shows the case index.
            if x_var == self._case_index_str:
                vmax = max_size
                cbar_label = 'Array Index'
            else:
                vmax = len(cases)
                cbar_label = self._case_index_str
            self._scalar_mappable.set_clim(0, vmax)
            self._colorbar.set_label(cbar_label)

    def _update_plot(self):
        """
        Update the plot based on the contents of the widgets.
        """
        with self._widgets['debug_output']:
            cr = self._case_reader
            src = self._widgets['source_select'].value
            cases = self._widgets['cases_list'].options
            x_var = self._widgets['x_select'].value
            y_var = self._widgets['y_select'].value
            x_slice = '' if self._widgets['x_slice'].value == '[...]' else self._widgets['x_slice'].value
            y_slice = '' if self._widgets['y_slice'].value == '[...]' else self._widgets['y_slice'].value
            x_transform = self._widgets['x_transform_select'].value
            y_transform = self._widgets['y_transform_select'].value
            x_var_type = self._widgets['x_var_type'].value
            y_var_type = self._widgets['y_var_type'].value

            try:
                self._ax.clear()
            except AttributeError:
                return

            if not cases or not x_var or not y_var:
                print('Nothing to plot')
                return

            x_units = 'None' if x_var == self._case_index_str else _get_output_meta(cr, cases[0], x_var)['units']
            y_units = _get_output_meta(cr, cases[0], y_var)['units']

            self._redraw_plot()

            x_label = rf'{x_var}{x_slice}'
            y_label = rf'{y_var}{y_slice}'

            if x_var_type == 'residuals':
                x_label = f'$\mathcal{{R}}$({x_label})'

            if y_var_type == 'residuals':
                y_label = f'$\mathcal{{R}}$({y_label})'

            if x_transform != 'None':
                x_label = f'{x_transform}({x_label})'

            if y_transform != 'None':
                y_label = f'{y_transform}({y_label})'

            self._ax.set_xlabel(f'{x_label}\n({x_units})')
            self._ax.set_ylabel(f'{y_label}\n({y_units})')
            self._ax.grid(True)

            self._ax.set_xscale(self._widgets['x_scale'].value)
            self._ax.set_yscale(self._widgets['y_scale'].value)
