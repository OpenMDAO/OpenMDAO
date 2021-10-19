import json
import operator
from functools import reduce

from bokeh.io import show, output_notebook
from bokeh.models import Select, HoverTool, MultiSelect
from bokeh.layouts import row, column
from bokeh.plotting import figure, ColumnDataSource
from bokeh.palettes import Category20, Turbo256

from openmdao.utils.notebook_utils import notebook
import openmdao.api as om
from openmdao.utils.om_warnings import issue_warning

import numpy as np


class VarOptViewer(object):

    def __init__(self, data, port=8888):
        """
        Initialize threading.

        port : int
            What port to host Bokeh server on.
        data : CaseRecorder or str
            A path to the recorder file or CaseRecorder.
            Currently only sqlite database files recorded via SqliteRecorder are supported.
        """
        if not notebook:
            raise RuntimeError("OptView must be run in a notebook environment")

        self.data = data

        self.circle_data = ColumnDataSource(dict(x_vals=[], y_vals=[], color=[], cases=[]))
        self.multi_line_data = ColumnDataSource(dict(x_vals=[], y_vals=[], color=[], cases=[]))

        output_notebook()

        show(self._make_plot, notebook_handle=True, notebook_url=("http://localhost:" + str(port)))

    def _parse_cases(self):
        if isinstance(self.data, str):
            self.cr = om.CaseReader(self.data)
        if isinstance(self.data, om.SqliteRecorder):
            self.cr = self.data

    def var_hash_map(self, variables):

        var_hash = {}
        for variable in sorted(variables['outputs']):
            group = variable.split(".")[-1].split(":")[0]

            if group in var_hash:
                var_hash[group].append(variable)
            else:
                var_hash[group] = [variable]

        return var_hash

    def var_compatability_check(self, x_var, var_map):
        group = x_var.split('.')[-1].split(':')[0]

        if len(var_map[group]) > 1:
            compatible_vars = var_map[group].copy()
            compatible_vars.pop(compatible_vars.index(x_var))
            return compatible_vars + ["segment_length"]
        else:
            return ["segment_length"]

    def _parse(self):
        """
        Parse the case recorder.
        """
        opt_data = None

        self._parse_cases()
        cases = self.cr.get_cases()

        opt_data = {}
        for case in cases:
            if hasattr(case, 'opt_progress') and "{}" not in case.opt_progress:
                data = json.loads(case.opt_progress)
                for key, val in data.items():
                    if key not in opt_data:
                        opt_data[key] = [val]
                    else:
                        opt_data[key].append(val)

        return opt_data

    def _make_plot(self, doc):
        self._parse()

        self.doc = doc
        source_options = self.cr.list_sources(out_stream=None)
        self.case_options = [(str(i), case) for i, case in \
                             enumerate(self.cr.list_cases(source_options[0], out_stream=None))]
        self.io_options_x = self.cr.list_source_vars(source_options[0], out_stream=None)
        self.var_map = self.var_hash_map(self.io_options_x)
        for key in self.io_options_x:
            self.io_options_x[key].append("segment_length")

        for val in self.io_options_x.values():
            if val and val[0] != "segment_length":
                io_starting_option = val[0]
                break

        self.variables_plot = figure(title="Problem Variables", x_axis_label="Variable Length",
                                     y_axis_label="Variable X")

        self.variables_plot.circle(x="x_vals", y="y_vals", source=self.circle_data)
        line_plot = self.variables_plot.multi_line(xs="x_vals", ys="y_vals", line_width=2, line_color='color',
                                                   source=self.multi_line_data)

        self.source_select = Select(title="Source:", value=source_options[0],
                                    options=source_options)
        self.source_select.on_change('value', self._source_update)

        self.case_select = MultiSelect(title="Case:", value=["0"], options=self.case_options)
        self.case_select.on_change('value', self._case_select_update)
        self.case_select.height = 300

        self.io_select_x = Select(title="X Value:", value=io_starting_option, options=self.io_options_x)
        self.io_select_x.on_change('value', self._io_var_select_x_update)

        self.io_options_y = self.var_compatability_check(self.io_select_x.value, self.var_map)
        self.io_select_y = Select(title="Y Value:", value=self.io_options_y[0], options=self.io_options_y)
        self.io_select_y.on_change('value', self._io_var_select_y_update)

        self.variables_plot.yaxis.axis_label = self.io_options_y[0]
        self.variables_plot.xaxis.axis_label = io_starting_option

        self.layout = row(self.variables_plot, column(self.source_select,
                                                      self.case_select,
                                                      self.io_select_x,
                                                      self.io_select_y,
                                                      ))

        ht = HoverTool(renderers=[line_plot],
            tooltips=[
                ( 'Case',  '@cases')
            ],
            mode = 'mouse'
        )
        self.variables_plot.add_tools(ht)
        self.update()
        self.doc.add_root(self.layout)

    def _source_update(self, attr, old, new):
        self.case_select.options = [(str(i), case) for i, case in \
                                    enumerate(self.cr.list_cases(new, out_stream=None))]
        self.source_select.value = new
        self.case_select.value = ['0']
        self.update()

    def _case_select_update(self, attr, old, new):
        self.update()

    def _io_var_select_y_update(self, attr, old, new):
        self.variables_plot.yaxis.axis_label = new
        self.update()

    def _io_var_select_x_update(self, attr, old, new):


        if self.io_select_x.value == "segment_length":
            self.io_select_y.options = self.io_select_x.options
            self.variables_plot.xaxis.axis_label = new
            self.variables_plot.yaxis.axis_label = self.io_select_y.value
        else:
            self.io_select_y.options = self.var_compatability_check(new, self.var_map)
            self.io_select_y.value = self.io_select_y.options[0]

            self.variables_plot.xaxis.axis_label = new
            self.variables_plot.yaxis.axis_label = self.io_select_y.value

        self.update()

    def flatten_list(self, list_to_flatten):
        return reduce(operator.concat, list_to_flatten)

    def update(self):
        new_data = dict(
            x_vals=[],
            y_vals=[],
            color=[],
            cases=[]
        )

        for i in self.case_select.value:
            case = self.cr.get_case(self.case_options[int(i)][1])

            for key, val in self.io_options_x.items():
                if self.io_select_y.value in val:
                    y_io = getattr(case, key)

                if self.io_select_x.value in val:
                    x_io = getattr(case, key)

            if self.io_select_y.value == "segment_length" and \
                self.io_select_x.value == "segment_length":
                x_variable = list(range(1))
                y_variable = list(range(1))
            elif self.io_select_y.value == "segment_length":
                x_variable = x_io[self.io_select_x.value]
                y_variable = list(range(len(x_variable)))
            elif self.io_select_x.value == "segment_length":
                y_variable = y_io[self.io_select_y.value]
                x_variable = list(range(len(y_variable)))
            else:
                x_variable = x_io[self.io_select_x.value]
                y_variable = y_io[self.io_select_y.value]

            if isinstance(x_variable, np.ndarray):
                new_data['x_vals'].append(x_variable.flatten().tolist())
            else:
                new_data['x_vals'].append(x_variable)

            if isinstance(y_variable, np.ndarray):
                new_data['y_vals'].append(y_variable.flatten().tolist())
            else:
                new_data['y_vals'].append(y_variable)

        if len(new_data['x_vals'][0]) > 1:

            new_data['color'] = self._line_color_list(new_data['x_vals'])
            new_data['cases'] = [self.case_options[int(case)][1] for case in self.case_select.value]
            self.multi_line_data.data = new_data
            self.circle_data.data = {"x_vals": [], "y_vals": [], "color": [], "cases": []}
        else:
            for key, val in new_data.items():
                new_data[key] = self.flatten_list(val)
            self.circle_data.data = new_data
            self.multi_line_data.data = {"x_vals": [], "y_vals": [], "color": [], "cases": []}


    def _line_color_list(self, x_var_vals):

        length = len(x_var_vals)
        if length <= 3:
            colors = list(Category20[3])
            while len(colors)>length:
                colors.pop()
        else:
            if length > 20:
                print("Using larger color palette")
                colors = list(Turbo256[:length])
            elif length < 20:
                colors = list(Category20[length])
            else:
                issue_warning("Cannot compare more than 256 cases")

        return colors
