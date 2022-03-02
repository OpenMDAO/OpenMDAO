"""Widgets for accessing CaseReader in a Jupyter notebook."""

import ast
import pathlib
import re

import openmdao.api as om
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm

try:
    import ipywidgets as ipw
    from ipywidgets import interact, Layout
    from IPython.display import display
    from IPython import get_ipython
except Exception:
    widgets = None


_func_map = {'ravel': np.ravel,
            'min': np.min,
            'minabs': lambda x: np.min(np.abs(x)),
            'max': np.max,
            'maxabs': lambda x: np.max(np.abs(x)),
            'log10abs': lambda x: np.log10(np.abs(x)),
            'norm': np.linalg.norm}


def _apply_transform(data, transform):
    """
    Apply a function from _func_map to the data based on the specified transform.

    Parameters
    ----------

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
    safe_funcs = {'index_exp': np.index_exp}
    sl = eval(f'index_exp{s}', safe_funcs, None)
    return data[sl]

def _get_val_across_cases(cases, var, transform, sl):
    """
    Retrieve the value of variable across one or more cases.
    """
    shape = (len(cases),) + _apply_transform(cases[0].get_val(var), transform)[sl].shape
    out = np.zeros(shape)
    for i, case in enumerate(cases):
        out = _apply_transform(case.get_val(var), transform)[sl]
    return out


def _load_case_file(cr):
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
    outputs = {filename: {source_name: {} for source_name in cr.list_sources(out_stream=None)}}

    for source_name in outputs[filename].keys():
        outputs[filename][source_name] = {case: {} for case in cr.list_cases(source=source_name, out_stream=None)}

        for case_name in outputs[filename][source_name].keys():
            case_outputs = cr.get_case(case_name).list_outputs(prom_name=True, units=True, shape=True, val=False,
                                                               out_stream=None)
            outputs[filename][source_name][case_name] = {meta['prom_name']: meta for _, meta in case_outputs}

    return outputs


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

    Attributes
    ----------

    """

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

        # self._widgets['case_remove_all_button'] = ipw.Button(description='Remove All ' + 2*'\u274c',
        #                                                      layout={'width': '100%'})

        self._widgets['cases_list'] = ipw.Select(options=[default_case],
                                                 value=default_case,
                                                 layout=ipw.Layout(width='40%', height='auto'))

        self._widgets['x_filter'] = ipw.Text('', description='X-Axis Filter',
                                             layout=ipw.Layout(width='45%', height='auto'))

        self._widgets['x_select'] = ipw.Select(options=[self._case_index_str] + default_outputs_list,
                                               value=self._case_index_str,
                                               description='X-Axis',
                                               layout=ipw.Layout(width='45%', height='auto'))


        self._widgets['x_transform_select'] = ipw.Dropdown(options=_func_map.keys(),
                                                           value='ravel',
                                                           description='X Transform',
                                                           layout=ipw.Layout(width='95%', height='auto'))

        self._widgets['x_slice'] = ipw.Text('[...]', description='X Slice',
                                            layout=ipw.Layout(width='95%', height='auto'))

        self._widgets['x_info'] = ipw.HTML(value='', description='X Shape', layout={'width': '95%'})

        self._widgets['y_filter'] = ipw.Text('', description='Y-Axis Filter',
                                 layout=ipw.Layout(width='45%', height='auto'))

        self._widgets['y_select'] = ipw.Select(options=default_outputs_list,
                                   description='Y-Axis',
                                   layout=ipw.Layout(width='45%', height='auto'))

        self._widgets['y_transform_select'] = ipw.Dropdown(options=_func_map.keys(),
                                                           value='ravel',
                                                           description='Y Transform',
                                                           layout=ipw.Layout(width='95%', height='auto'))

        self._widgets['y_slice'] = ipw.Text('[...]',
                                description='Y Slice',
                                layout=ipw.Layout(width='95%', height='auto'))

        self._widgets['y_info'] = ipw.HTML(value='', description='Y Shape', layout={'width': '95%'})

        self._widgets['case_slider'] = ipw.IntSlider(value=1, min=0, max=1, step=1, description='Case #',
                                                     disabled=False, continuous_update=True, orientation='horizontal',
                                                     readout=True, readout_format='d', layout=Layout(width='95%'))

        self._widgets['debug_output'] = ipw.Output(description='',
                                                   layout={'border': '0px solid black', 'width': '95%',
                                                           'height': '400'})

        display(ipw.VBox([self._widgets['source_select'],
                          ipw.HBox([self._widgets['cases_select'], ipw.VBox([self._widgets['case_select_button'],
                                                                             self._widgets['case_select_all_button'],
                                                                             self._widgets['case_remove_button']]),
                                    self._widgets['cases_list']], layout={'width': '95%'}),
                          self._widgets['x_filter'],
                          ipw.HBox([self._widgets['x_select'], ipw.VBox([self._widgets['x_info'],
                                                                         self._widgets['x_slice'],
                                                                         self._widgets['x_transform_select'],
                                                                         ])]),
                          self._widgets['y_filter'],
                          ipw.HBox([self._widgets['y_select'], ipw.VBox([self._widgets['y_info'],
                                                                         self._widgets['y_slice'],
                                                                         self._widgets['y_transform_select'],
                                                                         ])]),
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

            self._widgets['x_select'].observe(self._callback_select_var, 'value')
            self._widgets['y_select'].observe(self._callback_select_var, 'value')

            self._widgets['x_slice'].observe(self._callback_change_slice, 'value')
            self._widgets['y_slice'].observe(self._callback_change_slice, 'value')

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
            # self._set_num_plot_lines(len(clw.options))

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
            s = w.value
            src = self._widgets['source_select'].value
            cases = self._widgets['cases_list'].options

            r = re.compile(s)
            filtered_list = list(filter(r.search, self.outputs(src, cases).keys()))

            if w is self._widgets['x_filter']:
                var_select = self._widgets['x_select']
                filtered_list = [self._case_index_str] + filtered_list
            else:
                var_select = self._widgets['y_select']

            var_select.options = filtered_list

    def _callback_select_var(self, *args):
        event = args[0]
        self._widgets['debug_output'].clear_output()
        with self._widgets['debug_output']:
            w = event['owner']
            s = w.value
            src = self._widgets['source_select'].value
            cases = self._widgets['cases_list'].options

            axis = 'x' if w is self._widgets['x_select'] else 'y'
            var_info = self._widgets[f'{axis}_info']

            if s == self._case_index_str:
                shape = (len(cases),)
            else:
                shape = self.outputs(src, cases)[s]['shape']
            var_info.value = str(shape)

            self._update_var_info(axis)
            self._update_plot()

    def _callback_change_slice(self, *args):
        event = args[0]

        self._widgets['debug_output'].clear_output()
        with self._widgets['debug_output']:
            w = event['owner']
            s = w.value

            if s.startswith('[') and s.endswith(']'):
                self._update_plot()

    def _callback_select_transform(self, *args):
        self._update_plot()

    def _update_plot_vs_case_index(self, cases, y_var):
        """
        Update the plot area by plotting the current y vs the case index.

        Parameters
        ----------
        cases
        y_var

        Returns
        -------

        """
        x_min = y_min = 1E16
        x_max = y_max = -1E16

        self._widgets['debug_output'].clear_output()
        with self._widgets['debug_output']:
            for i, case in enumerate(cases):
                y_val = self._case_reader.get_case(case).get_val(y_var)
                # TODO: Transform as needed
                x_val = i * np.ones_like(y_val)
                x_min = min(x_min, np.min(x_val))
                x_max = max(x_max, np.max(x_val))
                y_min = min(y_min, np.min(y_val))
                y_max = max(y_max, np.max(y_val))

                self._ax.plot(x_val, y_val,
                              color=self._cmap[i, ...] ,
                              marker='o',
                              linestyle='-',
                              linewidth=1,
                              markersize=5)

                self._fig.canvas.flush_events()

            x_margin = (x_max - x_min) * 0.05
            x_margin = 0.1 if x_margin < 1.0E-16 else x_margin
            y_margin = (y_max - y_min) * 0.05
            y_margin = 0.1 if y_margin < 1.0E-16 else y_margin
            self._ax.set_xlim(x_min - x_margin, x_max + x_margin)
            self._ax.set_ylim(y_min - y_margin, y_max + y_margin)

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

        selected_case_idx = self._widgets['case_slider'].value

        self._widgets['debug_output'].clear_output()
        with self._widgets['debug_output']:
            for i, case in enumerate(cases):
                alpha = 1.0 if (selected_case_idx >= len(cases) or i == selected_case_idx) else 0.2
                lw = 1.0 if (selected_case_idx >= len(cases) or i == selected_case_idx) else 0.05

                y_val = self._case_reader.get_case(case).get_val(y_var)

                try:
                    y_val = _apply_slice(y_val, y_slice)
                except:
                    print(f'Error while applying Y slice: {y_slice}')
                y_val = _apply_transform(y_val, y_transform)

                if x_var not in (None, self._case_index_str):
                    x_val = self._case_reader.get_case(case).get_val(x_var)
                    if x_val.shape[0] != y_val.shape[0]:
                        print(f'Incompatible shapes: x.shape = {x_val.shape}  y.shape = {y_val.shape}.')
                        print('Size along first axis must agree.')
                        return
                else:
                    x_val = i * np.ones_like(y_val)
                try:
                    x_val = _apply_slice(x_val, x_slice)
                except:
                    print(f'Error while applying X slice: {x_slice}')
                x_val = _apply_transform(x_val, x_transform)

                x_min = min(x_min, np.min(x_val))
                x_max = max(x_max, np.max(x_val))
                y_min = min(y_min, np.min(y_val))
                y_max = max(y_max, np.max(y_val))

                self._ax.plot(x_val, y_val,
                              color=self._cmap[i, ...],
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
            self._ax.set_xlim(x_min - x_margin, x_max + x_margin)
            self._ax.set_ylim(y_min - y_margin, y_max + y_margin)

    def _update_plot(self):
        self._widgets['debug_output'].clear_output()
        with self._widgets['debug_output']:
            src = self._widgets['source_select'].value
            cases = self._widgets['cases_list'].options
            x_var = self._widgets['x_select'].value
            y_var = self._widgets['y_select'].value


            self._ax.clear()

            self._cmap = cm.get_cmap('viridis', 256)(np.linspace(0, 1, len(cases)))

            x_units = 'None' if x_var == self._case_index_str else \
                self._outputs[self._filename][src][cases[0]][x_var]['units']
            y_units = self._outputs[self._filename][src][cases[0]][y_var]['units']

            self._redraw_plot(cases, x_var=x_var, y_var=y_var)

            self._ax.set_xlabel(f'{x_var} ({x_units})')
            self._ax.set_ylabel(f'{y_var} ({y_units})')
            self._ax.grid(True)


    def __init__(self, f, source=None, cases=None, x_axis=None, y_axis=None):
        """
        Initialize the case viewer interface.
        """
        get_ipython().run_line_magic('matplotlib', 'widget')

        self._case_reader = om.CaseReader(f) if isinstance(f, str) else f

        self._cmap = cm.get_cmap('viridis', 1)

        self._case_index_str = 'Case Index'

        self._filename = self._case_reader._filename

        self._outputs = _load_case_file(self._case_reader)

        self._make_gui()

        self._register_callbacks()

        self._fig, self._ax = plt.subplots(1, 1, figsize=(9, 9/1.6), tight_layout=True)

        self._update_var_info('x')
        self._update_var_info('y')

        return

#         all_cases_list = cr.list_cases('driver', out_stream=None)
#
#         for case_name in all_cases_list:
#             all_outputs = dict([(o[1]['prom_name'], o[1]) for o in
#                                 cr.get_case(case_name).list_outputs(prom_name=True, units=True, shape=True,
#                                                                     out_stream=None)])
#         self._all_outputs = all_outputs
#
#         self._case_index_label = 'Case Index'
#
#         self.w_source = ipw.Dropdown(options=self._sources,
#                                      value=self._sources[0] if self._sources else None,
#                                      description='Source',
#                                      layout=ipw.Layout(width='30%', height='auto'))
#
#         self.case_select = ipw.SelectMultiple(options=self._cases,
#                                               description='Cases',
#                                               layout=ipw.Layout(width='55%', height='auto'))
#
#         self.x_filter = ipw.Text('', description='X-Axis Filter',
#                                  layout=ipw.Layout(width='55%', height='auto'))
#
#         self.x_select = ipw.Select(options=[self._case_index_label] + self._format_vars(all_outputs, self.x_filter),
#                                    description='X-Axis',
#                                    layout=ipw.Layout(width='55%', height='auto'))
#
#         self.x_transform_select = ipw.Dropdown(options=['ravel', 'min', 'max', 'norm', 'slice'],
#                                                value='ravel',
#                                                description='X Transform',
#                                                layout=ipw.Layout(width='95%', height='auto'))
#
#         self.x_slice = ipw.Text('[...]',
#                                 description='X Slice',
#                                 disabled=True,
#                                 layout=ipw.Layout(width='95%', height='auto'))
#
#         self.y_filter = ipw.Text('', description='Y-Axis Filter',
#                                  layout=ipw.Layout(width='55%', height='auto'))
#
#         self.y_select = ipw.Select(options=self._format_vars(all_outputs, self.y_filter),
#                                    description='Y-Axis',
#                                    layout=ipw.Layout(width='55%', height='auto'))
#
#         self.y_transform_select = ipw.Dropdown(options=['ravel', 'min', 'max', 'norm', 'slice'],
#                                                value='ravel',
#                                                description='Y Transform',
#                                                layout=ipw.Layout(width='95%', height='auto'))
#
#         self.y_slice = ipw.Text('[...]',
#                                 description='Y Slice',
#                                 disabled=True,
#                                 layout=ipw.Layout(width='95%', height='auto'))
#
#         self.x_filter.observe(self._apply_x_filter, 'value')
#         self.y_filter.observe(self._apply_y_filter, 'value')
#         self.case_select.observe(self._update_plot, 'value')
#         self.x_select.observe(self._update_plot, 'value')
#         self.y_select.observe(self._update_plot, 'value')
#
#         self.debug_output = ipyw.Output(description='Info',
#                                         layout={'border': '1px solid black',
#                                                 'width': '65%',
#                                                 'height': '100px'})
#
#         display(ipw.VBox([self.w_source,
#                           self.case_select,
#                           self.x_filter,
#                           ipw.HBox([self.x_select, ipw.VBox([self.x_transform_select, self.x_slice])]),
#                           self.y_filter,
#                           ipw.HBox([self.y_select, ipw.VBox([self.y_transform_select, self.y_slice])]),
#                           ipw.HBox([ipw.Label(value='Info'), self.debug_output])]))
#
#
#         self._plot = figure(title="simple line example", height=600, width=800, y_range=(-5, 5),
#                             background_fill_color='#efefef')
#
#         self._plot.line([], [])
#
#         out = ipw.Output()
#         display(out)
#
#         self._plot_line = p.line([], [])
#
#         with out:
#             self._plot_handle = show(self._plot, notebook_handle=True)
#
#         # set initial values
#         if source:
#             w_source.value = source
#
#         if cases:
#             self.case_select.value = cases
#         else:
#             self.case_select.value = [self._cases[-1]]
#
#         if x_axis:
#             self.x_select.value = x_axis
#         if y_axis:
#             if isinstance(y_axis, str):
#                 y_axis = [y_axis, ]
#             self.y_select.value = y_axis
#
#     def _format_vars(self, var_dict, filter_widget):
#         r = re.compile(filter_widget.value)
#         filtered_list = list(filter(r.search, var_dict.keys()))
#         if filtered_list:
#             max_length = max([len(s) for s in filtered_list])
#         else:
#             max_length = 0
#         return [f'{var.ljust(max_length)}  {var_dict[var]["shape"]}' for var in filtered_list]
#
#     def _apply_x_filter(self, *args):
#         filter_widget = self.x_filter
#         var_widget = self.x_select
#         flat_outputs = [op for op in self._all_outputs if np.prod(op['shape']) == max(op['shape'])]
#         opts = self._format_vars(flat_outputs, self.x_filter)
#         opts.insert(0, self._case_index_label)
#         var_widget.options = opts
#
#     def _apply_y_filter(self, *args):
#         filter_widget = self.y_filter
#         var_widget = self.y_select
#         opts = self._format_vars(self._all_outputs, self.y_filter)
#         var_widget.options = opts
#
#     def _get_range_extents(self):
#         case_list = self.case_select.value
#         if not case_list:
#             return
#
#         y_var = self.y_select.value.split()[0]
#
#         x_min = 1E16
#         x_max = -1E16
#
#         y_min = 1E16
#         y_max = -1E16
#
#         if self.x_select.value == self._case_index_label:
#             x = len(case_list)
#             x_is_case = True
#             x_min = 0
#             x_max = len(case_list)
#         else:
#             x_name = self.x_select.value.split()[0]
#             x_is_case = False
#             for i, case_name in enumerate(case_list):
#                 case = self._case_reader.get_case(case_name)
#                 if not x_is_case:
#                     x = case.get_val(x_name).ravel()
#                     x_min = min(x_min, np.min(x))
#                     x_max = min(x_max, np.max(x))
#                 y = case.get_val(y_var).ravel()
#                 y_min = min(y_min, np.min(y))
#                 y_max = max(y_max, np.max(y))
#
#         return x_min, y_min, x_max, y_max
#
#     def _update_plot(self, *args):
#
#         with self.debug_output:
#             case_list = self.case_select.value
#             if not case_list:
#                 return
#
#             x_var = self.x_select.value.split()[0]
#             if not x_var:
#                 return
#
#             y_var = self.y_select.value.split()[0]
#             if not y_var:
#                 return
#
#             if self.x_select.value == self._case_index_label:
#                 x = len(case_list)
#                 x_is_case_index = True
#             else:
#                 x_name = self.x_select.value.split()[0]
#                 x_is_case_index = False
#
#             x_transform = self.x_transform_select.value
#             y_transform = self.y_transform_select.value
#
#             x_slice = np.s_(ast.literal_eval(self.x_slice.value)) if x_transform == 'slice' else np.s_[...]
#             y_slice = np.s_(ast.literal_eval(self.y_slice.value)) if y_transform == 'slice' else np.s_[...]
#
#             self._plot.renderers.clear()
#
#             n_cases = len(case_list)
#             cmap = viridis(n_cases)
#
#             cases = [self._case_reader.get_case(c) for c in case_list]
#
#             #         x_min, y_min, x_max, y_max = self._get_range_extents()
#
#             #         self._plot.x_range.start = x_min
#             #         self._plot.x_range.end = x_max
#
#             #         self._plot.y_range.start = y_min
#             #         self._plot.y_range.end = y_max
#
#             if x_is_case_index:
#                 ys = []
#                 for i, case in enumerate(cases):
#                     y = _apply_transform(case.get_val(y_var), y_transform)[y_slice]
#                     self._plot.circle(i * np.ones_like(y), y, color=cmap[i], line_width=1.5, alpha=1.0)
#
#             else:
#                 # plot x vs y across cases
#                 x = _get_val_across_cases(cases, x_var, x_transform, x_slice)
#                 y = _get_val_across_cases(cases, y_var, y_transform, y_slice)
#
#             try:
#                 self._plot.circle(x, y, color=cmap[0], line_width=1.5, alpha=1.0)
#             except:
#                 pass
#
#             push_notebook(handle=self._plot_handle)
#
#
#
#
#
#
#
#
#
#
#
# import numpy as np
#
# from openmdao.recorders.case_reader import CaseReader
# from openmdao.utils.general_utils import simple_warning
#

# class CaseViewerWidget(object):
#     """
#     Widget to plot data from a CaseReader.
#
#     Parameters
#     ----------
#     cr : CaseReader or str
#         CaseReader or path to the recorded data file.
#     source : str, optional
#         Initial value for source.
#     cases : 2-tuple of int
#         Initial value for cases.
#     x_axis : str, optional
#         Initial value for x_axis.
#     y_axis : str or list of str, optional
#         Initial value for y_axis.
#     """
#
#     def __init__(self, cr, source=None, cases=None, x_axis=None, y_axis=None):
#         """
#         Initialize.
#         """
#         if widgets is None:
#             simple_warning(f"ipywidgets is required to use {self.__class__.__name__}."
#                            "To install it run `pip install openmdao[notebooks]`.")
#             return
#
#         if plt is None:
#             simple_warning(f"matplotlib is required to use {self.__class__.__name__}."
#                            "To install it run `pip install openmdao[visualization]`.")
#             return
#
#         if isinstance(cr, str):
#             cr = CaseReader(cr)
#
#         w_source = widgets.Dropdown(
#             options=cr.list_sources(out_stream=None),
#             description='Source:',
#             disabled=False,
#             layout=Layout(width='50%')
#         )
#
#         w_cases = widgets.IntRangeSlider(
#             value=[0, 0],
#             min=0,
#             max=0,
#             step=1,
#             description='Cases',
#             disabled=False,
#             continuous_update=False,
#             orientation='horizontal',
#             readout=True,
#             readout_format='d',
#             layout=Layout(width='50%')
#         )
#
#         w_xaxis = widgets.Dropdown(
#             options=['Iterations'],
#             value='Iterations',
#             description='X Axis:',
#             disabled=False,
#             layout=Layout(width='50%')
#         )
#
#         w_yaxis = widgets.SelectMultiple(
#             options=cr.list_source_vars(w_source.value, out_stream=None)['outputs'],
#             rows=len(cr.list_source_vars(w_source.value, out_stream=None)['outputs']),
#             description='Y Axis',
#             disabled=False,
#             layout=Layout(width='50%')
#         )
#
#         def source_changed(*args):
#             # keep selected vars
#             oldvars = set(w_yaxis.value)
#
#             # temporarily set harmless values
#             w_cases.value = [0, 0]
#             w_yaxis.value = []
#
#             # update cases
#             last_case = len(cr.list_cases(w_source.value, recurse=False, out_stream=None)) - 1
#             w_cases.max = last_case
#             w_cases.value = [0, last_case]
#
#             # update yaxis
#             yaxis = cr.list_source_vars(w_source.value, out_stream=None)['outputs']
#             w_yaxis.options = yaxis
#             w_yaxis.rows = min(len(yaxis), 15)
#             w_yaxis.value = list(oldvars.intersection(set(yaxis)))
#
#         w_source.observe(source_changed, 'value')
#         source_changed()
#
#         def cases_changed(*args):
#             if w_cases.value[0] == w_cases.value[1]:
#                 w_xaxis.options = sorted(w_yaxis.options)
#                 # w_xaxis.value = w_yaxis.options[0]
#             else:
#                 w_xaxis.options = ['Iterations']
#                 w_xaxis.value = 'Iterations'
#
#         w_cases.observe(cases_changed, 'value')
#         cases_changed()
#
#         messages = widgets.Output()
#
#         @messages.capture(clear_output=True)
#         def plot_func(source, cases, xaxis, yaxis):
#             # check if selected cases are not yet in sync with new source
#             case_ids = cr.list_cases(source, recurse=False, out_stream=None)
#             if cases[1] > len(case_ids) - 1:
#                 return
#
#             # check if selected outputs are not yet in sync with new source
#             case_outputs = cr.list_source_vars(source, out_stream=None)['outputs']
#             if not (set(yaxis) <= set(case_outputs)):
#                 return
#
#             case_nums = list(range(cases[0], cases[1] + 1))
#
#             if xaxis == 'Iterations':
#                 x = case_nums
#                 selected_case_ids = [case_ids[n] for n in case_nums]
#                 for outvar in yaxis:
#                     y = [cr.get_case(case_id).outputs[outvar] for case_id in selected_case_ids]
#                     plt.plot(x, np.array(y), label=outvar)
#                 step = 1 if len(x) <= 15 else len(x) // 15
#                 plt.xticks(np.arange(cases[0], cases[1], step=step))
#                 plt.xlabel('Iterations')
#             else:
#                 selected_case = cr.get_case(case_ids[case_nums[0]])
#                 x = selected_case.outputs[xaxis]
#                 if len(x.shape) > 1:
#                     print("Output chosen for X axis must be one-dimensional, "
#                           f"but {xaxis} has shape {x.shape}")
#                     return
#                 for outvar in yaxis:
#                     y = selected_case.outputs[outvar]
#                     if y.shape[0] != x.shape[0]:
#                         print(f"{xaxis} and {outvar} must have same first dimension, "
#                               f"but have shapes {x.shape} and {y.shape}")
#                         continue
#                     plt.plot(x, np.array(y), label=outvar)
#                 plt.xticks(np.arange(x[0], x[-1]))
#                 plt.xlabel(xaxis)
#
#             plt.grid(True)
#
#             if yaxis:
#                 plt.legend(loc="best")
#
#         plt.rcParams["figure.figsize"] = (16, 9)
#
#         interact(plot_func, source=w_source, cases=w_cases, xaxis=w_xaxis, yaxis=w_yaxis,
#                  disabled=False)
#
#         display(messages)
#
#         # set initial values
#         if source:
#             w_source.value = source
#         if cases:
#             w_cases.value = cases
#         if x_axis:
#             w_xaxis.value = x_axis
#         if y_axis:
#             if isinstance(y_axis, str):
#                 y_axis = [y_axis, ]
#             w_yaxis.value = y_axis
