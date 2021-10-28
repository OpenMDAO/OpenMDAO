import json
import operator
from functools import reduce
import warnings

from bokeh.io import show, output_notebook
from bokeh.models import Select, HoverTool, MultiSelect
from bokeh.layouts import row, column
from bokeh.plotting import figure, ColumnDataSource
from bokeh.palettes import Category20, Turbo256
from bokeh.util.warnings import BokehUserWarning

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

        warnings.simplefilter(action='ignore', category=BokehUserWarning)
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

    def var_compatability_check(self, variables, var_to_comp):

        variables = list(set(variables['inputs'] + variables['outputs'] + variables['residuals']))
        var_list = []

        for variable in variables:
            if variable in self.case.outputs and len(self.case[variable].flatten()) == len(self.case[var_to_comp].flatten()):
                var_list.append(variable)

        if var_list:
            return sorted(var_list) + ["Number of Points"] + ["Case Iterations"]
        elif variables != ["Number of Points", "Case Iterations"]:
            return ["Number of Points", "Case Iterations"]

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

        for key in self.io_options_x:
            if self.io_options_x[key]:
                io_starting_option = self.io_options_x[key][0]
            self.io_options_x[key] = sorted(self.io_options_x[key])


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

        self.io_options_y = self.io_options_x
        self.io_select_y = Select(title="Y Value:", value=io_starting_option, options=self.io_options_x)
        self.io_select_y.on_change('value', self._io_var_select_y_update)

        self.variables_plot.yaxis.axis_label = self.variables_plot.xaxis.axis_label = io_starting_option

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

        self.io_select_y.options = self.var_compatability_check(self.io_options_x, self.io_select_x.value)
        for key, val in self.io_select_x.options.items():
            self.io_select_x.options[key] = val + ["Number of Points"] + ["Case Iterations"]

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
        if self.io_select_x.value == "Number of Points" or \
            self.io_select_x.value == "Case Iterations":
            self.io_select_y.options = self.io_select_x.options
            self.variables_plot.xaxis.axis_label = new
            self.variables_plot.yaxis.axis_label = self.io_select_y.value
        else:
            self.io_select_y.options = self.var_compatability_check(self.io_select_x.options, new)
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
        num_points_x = num_points_y = False
        case_iter_x = case_iter_y = False

        if self.io_select_y.value == "Number of Points":
            num_points_y = True
        if self.io_select_x.value == "Number of Points":
            num_points_x = True

        if self.io_select_y.value == "Case Iterations":
            case_iter_y = True
        if self.io_select_x.value == "Case Iterations":
            case_iter_x = True

        if case_iter_x or case_iter_y:
            if len(self.case_select.value) == 1:
                current_val = int(self.case_select.value[0])
                self.case_select.value.append(str(current_val + 1))

        for i in self.case_select.value:
            self.case = self.cr.get_case(self.case_options[int(i)][1])

            for key, val in self.io_options_x.items():
                if self.io_select_y.value in val:
                    y_io = getattr(self.case, key)

                if self.io_select_x.value in val:
                    x_io = getattr(self.case, key)

            if (num_points_y and num_points_x) or (case_iter_x and case_iter_y):
                x_variable = list(range(1))
                y_variable = list(range(1))
            elif (num_points_x and case_iter_y) or (num_points_y and case_iter_x):
                x_variable = list(range(1))
                y_variable = list(range(1))
                print("Cannot compare Number of Points to Case Iterations")
            elif num_points_y or case_iter_y:
                x_variable = x_io[self.io_select_x.value].flatten()
                y_variable = list(range(len(x_variable)))
            elif num_points_x or case_iter_x:
                y_variable = y_io[self.io_select_y.value].flatten()
                x_variable = list(range(len(y_variable)))
            else:
                x_variable = x_io[self.io_select_x.value].flatten()
                y_variable = y_io[self.io_select_y.value].flatten()

            if not isinstance(new_data['x_vals'], np.ndarray):
                new_data['x_vals'] = np.empty((0, len(x_variable)), float)
                new_data['y_vals'] = np.empty((0, len(y_variable)), float)

            new_data['x_vals'] = np.vstack((new_data['x_vals'], x_variable))
            new_data['y_vals'] = np.vstack((new_data['y_vals'], y_variable))

        if new_data['x_vals'].shape[1] > 1:
            x_len = new_data['x_vals'].shape[1]
            y_len = new_data['y_vals'].shape[1]
            new_data['color'] = self._line_color_list(new_data['x_vals'])
            new_data['cases'] = [self.case_options[int(case)][1] for case in self.case_select.value]
            case_len = len(new_data['cases'])

            # Move this check outside of the if loop and then make it work if a user picks number of
            # points for both x and y
            if case_iter_x:
                if len(new_data['cases']) == 1:
                    issue_warning("Select two or more cases")
                new_data['x_vals'] = np.full((x_len, case_len), [list(range(0, case_len))]).T.tolist()
                new_data['y_vals'] = new_data['y_vals'].tolist()

            elif case_iter_y:
                if len(new_data['cases']) == 1:
                    issue_warning("Select two or more cases")
                new_data['y_vals'] = np.full((y_len, case_len), [list(range(0, case_len))]).T.tolist()
                new_data['x_vals'] = new_data['x_vals'].tolist()
            else:
                new_data['x_vals'] = new_data['x_vals'].tolist()
                new_data['y_vals'] = new_data['y_vals'].tolist()

            # For debugging purposes only. Delete for final release.
            self.x_vals = new_data['x_vals']
            self.y_vals = new_data['y_vals']

            self.multi_line_data.data = new_data
            self.circle_data.data = {"x_vals": [], "y_vals": [], "color": [], "cases": []}
        else:
            for key, val in new_data.items():
                if len(val) > 1:
                    new_data[key] = self.flatten_list(val)
                else:
                    new_data[key] = val
            new_data['color'] = self._line_color_list(new_data['x_vals'])
            new_data['cases'] = [self.case_options[int(case)][1] for case in self.case_select.value]

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
                colors = list(Turbo256[:length])
            elif length < 20:
                colors = list(Category20[length])
            else:
                issue_warning("Cannot compare more than 256 cases")

        return colors

    def test_func(self):
        # For debugging purposes only. Delete for final release.
        return self.x_vals, self.y_vals
