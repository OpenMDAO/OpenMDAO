"""Widgets for accessing CaseReader in a Jupyter notebook."""

import ast
import pathlib
import re
import webbrowser

import openmdao.api as om
import numpy as np

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
        return _func_map[transform](data)
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

        self._load_case_file(self._case_reader)

        self._make_gui()

        self._register_callbacks()

        self._fig, self._ax = plt.subplots(1, 1, figsize=(9, 9/1.6), tight_layout=True)

        self._update_var_info('x')
        self._update_var_info('y')

    def sources(self):
        return self._outputs[self._filename] if self._filename in self._outputs else {}

    def cases(self, source):
        srcs = self.sources()
        return srcs[source] if source in srcs else {}

    def outputs(self, source, cases):
        available_cases = self.cases(source)
        _cases = [cases] if isinstance(cases, str) else cases
        op = {}
        for case in _cases:
            op.update(available_cases[case] if case in available_cases else {})
        return op

    def _load_case_file(self, cr):
        """
        Load a CaseRecorder file into a dictionary structure.

        Parameters
        ----------
        cr : om.CaseReader
            The case reader object to load.

        Returns
        -------
        dict
            A nested dictionary where the metadata of the outputs are keyed with the following layers:
            `outputs[filename][source_name][case_name][output_name]`
        """
        filename = cr._filename
        self._outputs = {filename: {source_name: {} for source_name in cr.list_sources(out_stream=None)}}
        self._desvars = set()
        self._constraints = set()
        self._objectives = set()
        self._resids = set()

        for source_name in self._outputs[filename].keys():
            self._outputs[filename][source_name] = {case: {} for case in
                                                    cr.list_cases(source=source_name, out_stream=None)}

            for case_name in self._outputs[filename][source_name].keys():
                case = cr.get_case(case_name)
                case_outputs = case.list_outputs(prom_name=True, units=True, shape=True, val=False, residuals=True,
                                                 out_stream=None)
                self._outputs[filename][source_name][case_name] = {meta['prom_name']: meta for _, meta in case_outputs}
                self._desvars |= case.get_design_vars().keys()
                self._constraints |= case.get_constraints().keys()
                self._objectives |= case.get_objectives().keys()
                self._resids |= {promname for promname, meta in self._outputs[filename][source_name][case_name].items()
                                 if isinstance(meta['resids'], str) and meta['resids'] != 'Not Recorded'}

    def _make_gui(self):
        default_source = list(self.sources().keys())[0]
        default_case = list(self.cases(default_source).keys())[-1]
        default_outputs_list = list(self.outputs(default_source, default_case).keys())

        self._widgets = {}

        self._widgets['source_select'] = ipw.Dropdown(options=self.sources().keys(),
                                                      value=default_source,
                                                      description='Source',
                                                      layout=ipw.Layout(width='30%', height='auto'))

        self._widgets['cases_select'] = ipw.SelectMultiple(options=self.cases(default_source).keys(),
                                                          value=[default_case],
                                                          description='Cases',
                                                          layout=ipw.Layout(width='40%', height='auto'))

        self._widgets['case_select_button'] = ipw.Button(description='Select ' + '\u27F6',
                                                         layout={'width': '100%'})

        self._widgets['case_select_all_button'] = ipw.Button(description='Select All ' + '\u27F9',
                                                             layout={'width': '100%'})

        self._widgets['case_remove_button'] = ipw.Button(description='Remove ' + '\u274c',
                                                         layout={'width': '100%'})

        self._widgets['cases_list'] = ipw.Select(options=[default_case],
                                                 value=default_case,
                                                 layout=ipw.Layout(width='40%', height='auto'))

        self._widgets['x_filter'] = ipw.Text('', description='X-Axis Filter',
                                             layout=ipw.Layout(width='49%', height='auto'))

        self._widgets['x_var_type'] = ipw.Dropdown(options=['outputs', 'optimization', 'residuals'],
                                                   description='X Var Type',
                                                   value='outputs',
                                                   layout={'width': '49%'})

        self._widgets['x_select'] = ipw.Select(options=[self._case_index_str] + default_outputs_list,
                                               value=self._case_index_str,
                                               description='X-Axis',
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

        self._widgets['y_var_type'] = ipw.Dropdown(options=['outputs', 'optimization', 'residuals'],
                                                   description='Y Var Type',
                                                   value='outputs',
                                                   layout={'width': '49%'})

        self._widgets['y_select'] = ipw.Select(options=default_outputs_list,
                                   description='Y-Axis',
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


    def _update_var_info(self, axis):
        """
        Updates the variable info displayed.

        Parameters
        ----------
        axis : str
            'x' or 'y' - case insensitive.
        """
        if axis.lower() not in ('x', 'y'):
            raise ValueError(f'Unknown axis: {axis}')

        src = self._widgets['source_select'].value
        cases = self._widgets['cases_list'].options

        var = self._widgets[f'{axis}_select'].value

        if var == self._case_index_str:
            shape = (len(cases),)
            units = None
        elif var is None:
            shape = 'N/A'
            units = 'N/A'
        else:
            meta = self._outputs[self._filename][src][cases[0]][var]
            shape = meta['shape']
            units = meta['units']

        self._widgets[f'{axis}_info'].value = f'{shape}'


    def _register_callbacks(self):
        self._widgets['debug_output'].clear_output()
        with self._widgets['debug_output']:

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

    def _callback_select_case(self, *args):
        """
        Add the selected case(s) to the chosen cases list.
        """
        self._widgets['debug_output'].clear_output()
        with self._widgets['debug_output']:
            clw = self._widgets['cases_list']
            current = clw.options
            new = self._widgets['cases_select'].value
            numeric_sorter = lambda case_name: (case_name.split('|')[0], int(case_name.split('|')[-1]))
            self._widgets['cases_list'].options = sorted(list(set(current + new)), key=numeric_sorter)
            self._update_plot()

    def _callback_case_list_select(self, *args):
        n = len(self._widgets['cases_list'].options)
        self._widgets['case_slider'].max = n
        self._widgets['case_slider'].value = n

    def _callback_case_slider(self, *args):
        self._update_plot()

    def _callback_select_all_cases(self, *args):
        """
        Add the selected case(s) to the chosen cases list.
        """
        self._widgets['debug_output'].clear_output()
        with self._widgets['debug_output']:
            clw = self._widgets['cases_list']
            current = clw.options
            new = self._widgets['cases_select'].options
            numeric_sorter = lambda case_name: (case_name.split('|')[0], int(case_name.split('|')[-1]))
            self._widgets['cases_list'].options = sorted(list(set(current + new)), key=numeric_sorter)
            self._update_plot()
            self._update_var_info('x')
            self._update_var_info('y')

    def _callback_remove_case(self, *args):
        """
        Removes the selected case from the chosen cases list widget.
        """
        self._widgets['debug_output'].clear_output()
        with self._widgets['debug_output']:
            clw = self._widgets['cases_list']
            new_list = list(clw.options)
            new_list.remove(clw.value)
            clw.options = new_list
            self._update_var_info('x')
            self._update_var_info('y')

    def _callback_filter_vars(self, *args):
        event = args[0]
        self._widgets['debug_output'].clear_output()
        with self._widgets['debug_output']:
            w = event['owner']

            axis = 'x' if w is self._widgets['x_filter'] or w is self._widgets['x_var_type'] else 'y'

            var_filter = self._widgets[f'{axis}_filter'].value
            var_select = self._widgets[f'{axis}_select'].value
            var_type = self._widgets[f'{axis}_var_type'].value

            src = self._widgets['source_select'].value
            cases = self._widgets['cases_list'].options

            all_outputs = self.outputs(src, cases).keys()

            r = re.compile(var_filter)
            filtered_list = list(filter(r.search, self.outputs(src, cases).keys()))

            if var_type == 'optimization':
                filtered_list = [s for s in filtered_list if
                                 (s in self._objectives or s in self._desvars or s in self._constraints)]
            elif var_type.startswith('resid'):
                filtered_list = [s for s in filtered_list if s in self._resids]

            self._widgets[f'{axis}_select'].options = [self._case_index_str] + filtered_list \
                if axis == 'x' else filtered_list

    def _callback_select_var(self, *args):
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
        event = args[0]
        self._widgets['debug_output'].clear_output()
        with self._widgets['debug_output']:
            w = event['owner']
            s = w.value

            if s.startswith('[') and s.endswith(']'):
                self._update_plot()

    def _callback_change_scale(self, *args):
        self._widgets['debug_output'].clear_output()
        with self._widgets['debug_output']:
            event = args[0]
            w = event['owner']
            if w is self._widgets['x_scale']:
                self._ax.set_xscale(w.value)
            else:
                self._ax.set_yscale(w.value)

    def _callback_select_transform(self, *args):
        self._update_plot()

    def _redraw_plot(self, cases, y_var, x_var=None):
        """
        Update the plot area by plotting one variable vs another over one or more cases.

        Parameters
        ----------
        cases
        y_var
        x_var

        Returns
        -------

        """
        x_min = y_min = 1E16
        x_max = y_max = -1E16

        x_slice = self._widgets['x_slice'].value
        y_slice = self._widgets['y_slice'].value

        x_transform = self._widgets['x_transform_select'].value
        y_transform = self._widgets['y_transform_select'].value

        x_var_type = self._widgets['x_var_type'].value
        y_var_type = self._widgets['y_var_type'].value

        selected_case_idx = self._widgets['case_slider'].value

        self._widgets['debug_output'].clear_output()
        with self._widgets['debug_output']:
            for i, case in enumerate(cases):
                alpha = 1.0 if (selected_case_idx >= len(cases) or i == selected_case_idx) else 0.2
                lw = 1.0 if (selected_case_idx >= len(cases) or i == selected_case_idx) else 0.05

                if True:
                    y_val = self._case_reader.get_case(case).get_val(y_var)
                elif y_var_type == 'residuals':
                    listed_op = self._case_reader.get_case(case).list_outputs(prom_name=True, residuals=True, val=False)
                    d = {meta['prom_name']: meta for _, meta in listed_op}
                    y_val = d[y_var]['resids']

                try:
                    y_val = _apply_slice(y_val, y_slice)
                except:
                    print(f'Error while applying Y slice: {y_slice}')
                y_val = _apply_transform(y_val, y_transform)

                if x_var not in (None, self._case_index_str):
                    x_val = self._case_reader.get_case(case).get_val(x_var)
                else:
                    x_val = i * np.ones_like(y_val)

                try:
                    x_val = _apply_slice(x_val, x_slice)
                except:
                    print(f'Error while applying X slice: {x_slice}')
                x_val = _apply_transform(x_val, x_transform)

                if x_val.shape[0] != y_val.shape[0]:
                    print(f'Incompatible shapes: x.shape = {x_val.shape}  y.shape = {y_val.shape}.')
                    print('Size along first axis must agree.')
                    return

                x_min = min(x_min, np.min(x_val))
                x_max = max(x_max, np.max(x_val))
                y_min = min(y_min, np.min(y_val))
                y_max = max(y_max, np.max(y_val))

                if x_var in (None, self._case_index_str):
                    self._ax.scatter(x_val, y_val, c=np.arange(x_val.size), s=20, cmap=self._cmap, alpha=alpha)
                else:
                    self._ax.plot(x_val, y_val,
                                  color=self._cmap(float(i)/len(cases)),
                                  marker='o',
                                  linestyle='-',
                                  linewidth=lw,
                                  markersize=5,
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

    def _update_plot(self):
        self._widgets['debug_output'].clear_output()
        with self._widgets['debug_output']:
            src = self._widgets['source_select'].value
            cases = self._widgets['cases_list'].options
            x_var = self._widgets['x_select'].value
            y_var = self._widgets['y_select'].value
            x_slice = '' if self._widgets['x_slice'].value == '[...]' else self._widgets['x_slice'].value
            y_slice = '' if self._widgets['y_slice'].value == '[...]' else self._widgets['y_slice'].value
            x_transform = self._widgets['x_transform_select'].value
            y_transform = self._widgets['y_transform_select'].value

            self._ax.clear()

            x_units = 'None' if x_var == self._case_index_str else \
                self._outputs[self._filename][src][cases[0]][x_var]['units']
            y_units = self._outputs[self._filename][src][cases[0]][y_var]['units']

            self._redraw_plot(cases, x_var=x_var, y_var=y_var)

            x_label = rf'{x_var}{x_slice}'
            y_label = rf'{y_var}{y_slice}'

            if x_transform != 'None':
                x_label = f'{x_transform}({x_label})'

            if y_transform != 'None':
                y_label = f'{y_transform}({y_label})'

            self._ax.set_xlabel(f'{x_label}\n({x_units})')
            self._ax.set_ylabel(f'{y_label}\n({y_units})')
            self._ax.grid(True)

            self._ax.set_xscale(self._widgets['x_scale'].value)
            self._ax.set_yscale(self._widgets['y_scale'].value)
