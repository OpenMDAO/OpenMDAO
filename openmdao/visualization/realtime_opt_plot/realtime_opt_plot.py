"""A real-time plot monitoring the analysis driver process as an OpenMDAO script runs."""

import ctypes
import errno
import os
import sys
from collections import defaultdict
import sqlite3
from itertools import product
import time
from datetime import datetime, timedelta

try:
    from bokeh.models import (
        ColumnDataSource,
        CheckboxGroup,
        Button,
        Row,
        Div,
        BasicTicker,
        LinearColorMapper,
        ColorBar,
        Select,
        Column,
        ScrollBox,
        InlineStyleSheet,
    )
    from bokeh.layouts import gridplot, Spacer
    from bokeh.transform import transform
    from bokeh.plotting import figure
    from bokeh.server.server import Server
    from bokeh.application.application import Application
    from bokeh.application.handlers import FunctionHandler
    from bokeh.palettes import Viridis256

    bokeh_available = True
except ImportError:
    bokeh_available = False

import numpy as np

from openmdao.recorders.sqlite_reader import SqliteCaseReader
from openmdao.recorders.case import Case

try:
    from openmdao.utils.gui_testing_utils import get_free_port
except ImportError:
    # If get_free_port is unavailable, the default port will be used
    def get_free_port():
        return 5000


# Define Constants

# the time between calls to the udpate method
# if this is too small, the GUI interactions get delayed because
# code is busy trying to keep up with the periodic callbacks
_time_between_callbacks_in_ms = 1000

# Number of milliseconds for unused session lifetime
_unused_session_lifetime_milliseconds = 1000 * 60 * 10

# layout and format params
_left_side_column_width = 500
_analysis_driver_progress_box_font_size = 16
# how big the individual plots should be in the grid
_grid_plot_height_and_width = 240

style = InlineStyleSheet(css="""
:host(.div_header) {
    font-size: 20px;
    font-weight: bold;
}
""")

# some models have a large number of variables. Putting them all in a plot
#   is not practical. Initially show no more than this number
_max_number_initial_visible_sampled_variables = 3
# number of histogram bins in the histogram plots for the sampled variables
_num_histogram_bins = 30 

# variable names can be very long and too large to show completely
#  in the plot axes. This is the largest number of characters that
#  can be shown in the axes. The rest are elided. Full variable
#  shown as a tooltip
_max_label_length = 25
_elide_string = "..."

# palette for the color bar
_color_palette = Viridis256

# the color bar showing response needs an initial value before new data comes in
_initial_response_range_for_plots = (0,100)

# function to create the labels for the plots, need to elide long variable names
# include the units, if given
def _elide_variable_name_with_units(variable_name, units):
    if units:
        un_elided_string_length = len(f"{variable_name} ({units})")
    else:
        un_elided_string_length = len(variable_name)
    chop_length = max(un_elided_string_length - _max_label_length + len(_elide_string),0)
    if chop_length:
        variable_name = _elide_string + variable_name[chop_length:]
    if units:
        return f"{variable_name} ({units})"
    else:
        return variable_name

# Alternative: Access objects by ID
def access_by_id(doc, object_id):
    """Retrieve a Bokeh object by its ID"""
    try:
        for obj in doc.roots[0].references():
            if obj.id == object_id:
                return obj
    except IndexError as e:
        return None
    return None

def _is_process_running(pid):
    if sys.platform == "win32":
        # PROCESS_QUERY_LIMITED_INFORMATION is available on Windows Vista and later.
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

        # Attempt to open the process.
        handle = ctypes.windll.kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, pid
        )
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        else:
            # If OpenProcess fails, check if it's due to access being denied.
            ERROR_ACCESS_DENIED = 5
            if ctypes.windll.kernel32.GetLastError() == ERROR_ACCESS_DENIED:
                return True
            return False
    else:
        try:
            os.kill(pid, 0)
        except OSError as err:
            if err.errno == errno.ESRCH:  # No such process
                return False
            elif err.errno == errno.EPERM:  # Process exists, no permission to signal
                return True
            else:
                raise
        else:
            return True


def _realtime_opt_plot_setup_parser(parser):
    """
    Set up the realtime plot subparser for the 'openmdao opt_plot' command.

    Parameters
    ----------
    parser : argparse subparser
        The parser we're adding options to.
    """
    parser.add_argument(
        "case_recorder_filename",
        type=str,
        help="Name of openmdao case recorder filename. It should contain driver cases",
    )

    parser.add_argument(
        "--pid",
        type=int,
        default=None,
        help="Process ID of calling optimization script, "
        "defaults to None if called by the user directly",
    )
    parser.add_argument(
        "--no-display",
        action="store_false",
        dest="show",
        help="Do not launch browser showing plot. Used for CI testing",
    )


def _realtime_opt_plot_cmd(options, user_args):
    """
    Run the opt_plot command.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.
    user_args : list of str
        Args to be passed to the user script.
    """
    if bokeh_available:
        realtime_opt_plot(
            options.case_recorder_filename,
            _time_between_callbacks_in_ms,
            options.pid,
            options.show,
        )
    else:
        print(
            "The bokeh library is not installed so the real-time optimizaton "
            "lot is not available. "
        )
        return


class _CaseRecorderTracker:
    """
    A class that is used to get information from a case recorder.

    These methods are not provided by the SqliteCaseReader class.
    """

    def __init__(self, case_recorder_filename):
        self._case_recorder_filename = case_recorder_filename
        self._cr = None
        self._initial_case = (
            None  # need the initial case to get info about the variables
        )
        self._next_id_to_read = 1

    def get_case_reader(self):
        return self._cr

    def _open_case_recorder(self):
        if self._cr is None:
            self._cr = SqliteCaseReader(self._case_recorder_filename)

    def _get_case_by_counter(self, counter):
        # use SQL to see if a case with this counter exists
        with sqlite3.connect(self._case_recorder_filename) as con:
            con.row_factory = sqlite3.Row
            cur = con.cursor()
            cur.execute(
                "SELECT * FROM driver_iterations WHERE " "counter=:counter",
                {"counter": counter},
            )
            row = cur.fetchone()
        con.close()

        # use SqliteCaseReader code to get the data from this case
        if row:
            # TODO would be better to not have to open up the file each time
            self._open_case_recorder()
            var_info = self._cr.problem_metadata["variables"]
            case = Case(
                "driver",
                row,
                self._cr._prom2abs,
                self._cr._abs2prom,
                self._cr._abs2meta,
                self._cr._conns,
                var_info,
                self._cr._format_version,
            )

            return case
        else:
            return None

    def _get_data_from_case(self, driver_case):
        objs = driver_case.get_objectives(scaled=False)
        design_vars = driver_case.get_design_vars(scaled=False)
        constraints = driver_case.get_constraints(scaled=False)

        new_data = {
            "counter": int(driver_case.counter),
        }

        # get objectives
        objectives = {}
        for name, value in objs.items():
            objectives[name] = value
        new_data["objs"] = objectives

        # get des vars
        desvars = {}
        for name, value in design_vars.items():
            desvars[name] = value
        new_data["desvars"] = desvars

        # get cons
        cons = {}
        for name, value in constraints.items():
            cons[name] = value
        new_data["cons"] = cons

        return new_data

    def _get_new_case(self):
        # get the next unread case from the recorder
        driver_case = self._get_case_by_counter(self._next_id_to_read)
        if driver_case is None:
            return None

        if self._initial_case is None:
            self._initial_case = driver_case

        self._next_id_to_read += 1

        return driver_case

    def _get_obj_names(self):
        obj_vars = self._initial_case.get_objectives()
        return obj_vars.keys()

    def _get_desvar_names(self):
        design_vars = self._initial_case.get_design_vars()
        return design_vars.keys()

    def _get_cons_names(self):
        cons = self._initial_case.get_constraints()
        return cons.keys()

    def _get_units(self, name):
        try:
            units = self._initial_case._get_units(name)
        except RuntimeError as err:
            if str(err).startswith("Can't get units for the promoted name"):
                return "Ambiguous"
            raise
        except KeyError:
            return "Unavailable"

        if units is None:
            units = "Unitless"
        return units

    def _get_shape(self, name):
        item = self._initial_case[name]
        return item.shape


class _RealTimeOptPlot(object):
    """
    A class that handles all of the real-time plotting.

    Parameters
    ----------
    case_recorder_filename : str
        The path to the case recorder file.
    callback_period : double
        The time between Bokeh callback calls (in seconds).
    doc : bokeh.document.Document
        The Bokeh document which is a collection of plots, layouts, and widgets.
    pid_of_calling_script : int or None
        The process ID of the process that called the command to start the realtime plot.

        None if the plot was called for directly by the user.
    """

    def __init__(
        self, case_recorder_filename, callback_period, doc, pid_of_calling_script
    ):
        """
        Construct and initialize _RealTimeOptPlot instance.
        """
        self._case_recorder_filename = case_recorder_filename
        self._case_tracker = _CaseRecorderTracker(case_recorder_filename)
        self._pid_of_calling_script = pid_of_calling_script
        self._doc = doc

        # lists of the sampled variables in the analysis and the responses
        self._sampled_variables = None
        self._num_sampled_variables = None
        self._prom_responses = None

        # A dict indicating which sampled variables are visible in the grid
        #   based on what the user selected
        self._sampled_variables_visibility = {}

        # The current response being plotted
        self._prom_response = None

        # data source items for doing streaming in bokeh
        self._source = None
        self._source_stream_dict = {}
        self._hist_source = {}

        # used to keep track of the y min and max of the data so that
        # the axes ranges can be adjusted as data comes in
        self._prom_response_min = defaultdict(lambda: float("inf"))
        self._prom_response_max = defaultdict(lambda: float("-inf"))

        # the actually scatter plots. Need access to them to change the
        #  fill color for the dots when the response variable range changes
        #  due to new data.
        self._scatter_plots = []

        # dictionaries for the items in the grid.
        #  used to turn their visibility on and off
        self._scatter_plots_figure = {}
        self._hist_figures = {}
        self._row_labels = {}
        self._column_labels = {}

        # data used to populate the Analysis Driver Progress block in the plot
        self._start_time = time.time()
        self._num_samples_plotted = 0

        # this is the main method of the class. It gets called periodically by Bokeh
        # It looks for new data and if found, updates the plot with the new data
        def _update():
            new_case = self._case_tracker._get_new_case()
            if not new_case:
                return

            # See if Bokeh source object is defined yet. If not, set it up
            # since now we have data from the case recorder with info about the
            # variables to be plotted.
            # Also setup the overall page
            if self._source is None:
                self._setup_data_source()
                self._setup_figure()
                doc.add_root(self._overall_layout)

            # TODO - is the case of new_case None handled correctly?
            self._update_source_stream(new_case)
            self._update_scatter_plots(new_case)
            self._update_histograms()
            self._update_analysis_driver_progress_text_box()
            # end of _update method

        doc.add_periodic_callback(_update, callback_period)
        doc.title = "OpenMDAO Analysis Driver Progress Plot"

    def _setup_data_source(self):
        self._source_dict = {}

        responses = list(
            self._case_tracker.get_case_reader().problem_metadata["responses"].keys()
        )

        # convert to promoted names
        self._prom_responses = []
        for response in responses:
            if response in self._case_tracker.get_case_reader()._abs2prom["output"]:
                self._prom_responses.append(
                    self._case_tracker.get_case_reader()._abs2prom["output"][response]
                )
            else:
                raise RuntimeError(f"No prom for abs variable {response}")

        if not self._prom_responses:
            raise RuntimeError(f"Need at least one response variable.")

        outputs = self._case_tracker.get_case_reader().list_source_vars("driver", \
            out_stream=None)["outputs"]

        # Don't include response variables in sampled variabales.
        # Also, split up the remaining sampled variables into scalars and
        # non-scalars
        self._sampled_variables = []
        self._sampled_variables_non_scalar = []
        for varname in outputs:
            if varname not in self._prom_responses:
                shape = self._case_tracker._get_shape(varname)
                if shape == () or shape == (1,): # is scalar ??
                    self._sampled_variables.append(varname)
                else:
                    self._sampled_variables_non_scalar.append(varname)
        self._num_sampled_variables = len(self._sampled_variables)

        # want them sorted in the Sampled Variables selection box
        self._sampled_variables.sort()
        self._sampled_variables_non_scalar.sort()

        # setup the source
        for response in self._prom_responses:
            self._source_dict[response] = []
        for sampled_variable in self._sampled_variables:
            self._source_dict[sampled_variable] = []
        self._source = ColumnDataSource(self._source_dict)

    def _setup_figure(self):
    
        analysis_progress_box = self._make_analysis_driver_box()
    
        # Initially the response variable plotted is the first one
        self._prom_response = self._prom_responses[0]

        self._set_initial_number_initial_visible_sampled_variables()
        visible_variables = [var for var in self._sampled_variables if self._sampled_variables_visibility[var]]

        # Create a color mapper using Viridis (colorblind-friendly)
        self._color_mapper = LinearColorMapper(palette=_color_palette, 
                                               low=_initial_response_range_for_plots[0], 
                                               high=_initial_response_range_for_plots[1])

        color_bar = self._make_color_bar()

        sampled_variables_box = self._make_sampled_variables_box(color_bar)

        plots_and_labels_in_grid = []
        self._make_plots(plots_and_labels_in_grid)
        self._make_plot_labels(visible_variables, plots_and_labels_in_grid)
        grid_of_plots = gridplot(
            plots_and_labels_in_grid,
            ncols=self._num_sampled_variables+1,  # need one extra row and column in the grid for the axes labels
            toolbar_location=None,
        )

        self._make_overall_layout(analysis_progress_box, sampled_variables_box, color_bar, grid_of_plots)

    def _update_source_stream(self, new_case):
        # fill up the stream dict with the values.
        # These are fed to the bokeh source stream
        for response in self._prom_responses:
            self._source_stream_dict[response] = \
                new_case.get_val(response)[:1]
        for sampled_variable in self._sampled_variables:
            self._source_stream_dict[sampled_variable] = \
                new_case.get_val(sampled_variable)[:1]
        self._source.stream(self._source_stream_dict)

    def _update_scatter_plots(self, new_case):
        # update the min and max for the response variable
        for response in self._prom_responses:
            self._prom_response_min[response] = min(
                self._prom_response_min[response], new_case.get_val(response)[:1][0]
            )
            self._prom_response_max[response] = max(
                self._prom_response_max[response], new_case.get_val(response)[:1][0]
            )

        # update the color mapper that is used to color the dots in the scatter plots
        self._color_mapper.low = self._prom_response_min[self._prom_response]
        self._color_mapper.high = self._prom_response_max[self._prom_response]

    def _update_histograms(self):
        for sampled_variable in self._sampled_variables:
            x_data = self._source.data[sampled_variable]

            hist, edges = np.histogram(
                x_data, bins=_num_histogram_bins, range=(np.min(x_data), np.max(x_data))
            )
            self._hist_source[sampled_variable].data.update(
                {
                    "top": hist,
                    "left": edges[:-1],  # left edge of each bin
                    "right": edges[1:],  # right edge of each bin
                }
            )

            self._hist_figures[sampled_variable].x_range.start = np.min(x_data)
            self._hist_figures[sampled_variable].x_range.end = np.max(x_data)

    def _update_analysis_driver_progress_text_box(self):
        self._num_samples_plotted += 1
        last_updated_time_formatted = datetime.now().strftime("%H:%M:%S on %B %d, %Y")
        elapsed_total_time = time.time() - self._start_time
        elapsed_total_time_formatted = str(timedelta(seconds=int(elapsed_total_time)))
        self._analysis_driver_progress_text_box.text = f"""<div style="padding: 10px; font-size: {_analysis_driver_progress_box_font_size}px ">
                        <p>Script: {self._case_recorder_filename}</p>
                        <p>Number of samples: {self._num_samples_plotted}</p>
                        <p>Last updated: {last_updated_time_formatted}</p>
                        <p>Elapsed time: {elapsed_total_time_formatted}</p>
                        </div>"""

    def _set_initial_number_initial_visible_sampled_variables(self):
        number_initial_visible_sampled_variables = 0
        for sampled_var in self._sampled_variables:
            if number_initial_visible_sampled_variables < _max_number_initial_visible_sampled_variables:
                self._sampled_variables_visibility[sampled_var] = True
                number_initial_visible_sampled_variables += 1
            else:
                self._sampled_variables_visibility[sampled_var] = False

    def _make_sampled_variables_box(self, color_bar):
        # Make all the checkboxes for the Sample Variables area to the left of the plot
        #   that lets the user select what to plot. Also include the non scalar
        #   variables at the bottom of this box
        
        # header for the scalar Sampled Variables list
        sampled_variables_label = Div(
            text="Sampled Variables",
            stylesheets = [style], css_classes = ['div_header']
        )

        # make the checkbox group that contains the scalar sampled variables that can be
        #   turned off and on
        sampled_variable_active_index_list = []
        for i, sampled_var in enumerate(self._sampled_variables):
            if self._sampled_variables_visibility[sampled_var]:
                sampled_variable_active_index_list.append(i)

        sampled_variables_checkbox_group = CheckboxGroup(
            labels=self._sampled_variables,
            active=sampled_variable_active_index_list,
            width=_left_side_column_width
        )
        
        def _sampled_variable_checkbox_callback(attr, old, new):
            # old and new are lists in terms of index into the checkboxes
            # starts at 0
            # Find which checkbox was toggled by comparing old and new states
            added = set(new) - set(old)
            removed = set(old) - set(new)
            
            # added and removed should really only be lists of length 0 or 1
            #  since you can only check or uncheck one at a time
            if added:
                active = True
                var_name = self._sampled_variables[added.pop()]
                
            if removed:
                active = False
                var_name = self._sampled_variables[removed.pop()]
                
            # turn on or off visibility of histograms, scatter plots and labels
            self._sampled_variables_visibility[var_name] = active
            self._hist_figures[var_name].visible = active
            self._row_labels[var_name].visible = active
            self._column_labels[var_name].visible = active

            for i, (y, x) in enumerate(
                product(self._sampled_variables, self._sampled_variables)
            ):
                icolumn = i % self._num_sampled_variables
                irow = i // self._num_sampled_variables
                # only do the lower half
                if x != y:
                    self._scatter_plots_figure[(x,y)].visible = (icolumn < irow \
                        and self._sampled_variables_visibility[x] and self._sampled_variables_visibility[y] )
        
        sampled_variables_checkbox_group.on_change('active', _sampled_variable_checkbox_callback )

        # Add custom CSS that targets the Bokeh checkbox styling more specifically
        custom_css = """
        .bk-input-group .bk-checkbox input[type="checkbox"]:checked {
            background-color: #4CAF50 !important;
            border-color: #4CAF50 !important;
        }

        .bk-checkbox input[type="checkbox"]:checked {
            background-color: #4CAF50 !important;
            border-color: #4CAF50 !important;
            accent-color: #2E7D32 !important;
        }

        input[type="checkbox"]:checked {
            background-color: #4CAF50 !important;
            border-color: #4CAF50 !important;
            accent-color: #2E7D32 !important;
        }

        input[type="checkbox"]:checked::after {
            color: white !important;
        }

        .bk-checkbox input[type="checkbox"]:checked::after {
            color: white !important;
        }


        .bk-input-group label {
            font-size: 20px !important;
        }

        .bk-checkbox label {
            font-size: 20px !important;
        }
        """
        sampled_variables_checkbox_group.stylesheets = [custom_css]

        # Create the non scalar variables list for the GUI
        sampled_variables_non_scalar_label = Div(
            text="Sampled Variables Non Scalar",
            stylesheets = [style], css_classes = ['div_header']
        )
        
        sampled_variables_non_scalar_text_list = []
        for sampled_var in self._sampled_variables_non_scalar:
            sampled_variables_non_scalar_text = Div(text=sampled_var,
                                                                styles={"font-size": "20px"},
        )           
            sampled_variables_non_scalar_text_list.append(sampled_variables_non_scalar_text)
        sampled_variables_non_scalar_column = Column(
            children=sampled_variables_non_scalar_text_list,
        )
      
        # put both the scalar and non scalar together in a Column inside a ScrollBox
        sampled_variables_column = Column(
            sampled_variables_label,
            sampled_variables_checkbox_group,
            sampled_variables_non_scalar_label,
            sampled_variables_non_scalar_column,
            sizing_mode="stretch_height",
            height_policy="fit",
        )
        sampled_variables_box = ScrollBox(
            child=sampled_variables_column,
            sizing_mode="stretch_height",
            height_policy="max",
        )
        
        # make response variable UI
        response_variable_menu = Select(
            options=self._prom_responses,
            value=self._prom_responses[0],  # Default value
        )
        response_variable_header = Div(
            text="Response variable",
            stylesheets = [style], css_classes = ['div_header']
        )

        def cb_select_response_variable(color_bar):
            def toggle_callback(attr, old, new):
                self._prom_response = new 
                color_bar.title = f"Response variable: '{new}'"
                self._color_mapper.low = self._prom_response_min[self._prom_response]
                self._color_mapper.high = self._prom_response_max[self._prom_response]

                for scatter_plot in self._scatter_plots:
                    scatter_plot.glyph.fill_color = transform(
                            self._prom_response, self._color_mapper
                        )
            return toggle_callback

        response_variable_menu.on_change("value", cb_select_response_variable(color_bar))

        sampled_variables_box = Column(
            response_variable_header,
            response_variable_menu,
            Spacer(height=20),
            sampled_variables_box,
            sizing_mode="stretch_height",
            height_policy="fit",
            width=_left_side_column_width,
            styles={
                'max-height': '100vh',  # Ensures it doesn't exceed viewport
                'border-radius': '5px',
                'border': '5px solid black',
            },
        )
               
        return sampled_variables_box

    def _make_color_bar(self):
        # Add the color bar to this figure
        color_bar = ColorBar(
            color_mapper=self._color_mapper,
            title=f"Response variable: '{self._prom_response}'",
            border_line_color=None,
            width=20,
            label_standoff=14,
            title_text_font_size="20px",
            ticker=BasicTicker(),
            location=(0, 0),
        )

        # Need the color bar to be associated with a plot figure.
        # So make a basic one and hide it
        p = figure(height=2 * _grid_plot_height_and_width, width=0, toolbar_location=None)
        # Plot a line between the two points
        line = p.line([0, 1],[0, 1])
        p.grid.grid_line_color = None
        line.visible = False
        p.xaxis.visible = False
        p.yaxis.visible = False
        p.xgrid.visible = False
        p.ygrid.visible = False
        p.title.visible = False
        p.outline_line_alpha = 0
        p.min_border_right = 100

        p.add_layout(color_bar, "right")

        return p

    def _make_plots(self, plots_and_labels_in_grid):
        # x and y here refer to the x axis and y axis of the grid
        # Add plots by row starting from top
        for i, (var_along_columns,var_along_rows) in enumerate(
            product(self._sampled_variables, self._sampled_variables)
        ):
            icolumn = i % self._num_sampled_variables
            irow = i // self._num_sampled_variables

            if var_along_columns != var_along_rows:
                p = figure(
                    background_fill_color="#fafafa",
                    border_fill_color="white",
                    width=_grid_plot_height_and_width,
                    height=_grid_plot_height_and_width,
                    output_backend="webgl",
                )
                self._scatter_plots_figure[(var_along_columns,var_along_rows)] = p
                p.axis.visible = True
                self._scatter_plots.append(p.scatter(
                    x=var_along_columns,
                    source=self._source,
                    y=var_along_rows,
                    size=5,
                    line_color=None,
                    fill_color=transform(
                        self._prom_response, self._color_mapper
                    ),  # This maps value ofself._prom_response variable to colors
                ))
            else:  # on the diagonal
                # Extract the x column data for the histogram
                x_data = self._source.data[var_along_columns]

                hist, edges = np.histogram(x_data, bins=_num_histogram_bins)
                # Create a new ColumnDataSource for the histogram data
                self._hist_source[var_along_columns] = ColumnDataSource(
                    data={
                        "top": hist,  # height of each bin
                        "bottom": np.zeros(len(hist)),  # bottom of each bin starts at 0
                        "left": edges[:-1],  # left edge of each bin
                        "right": edges[1:],  # right edge of each bin
                    }
                )
                p = figure(
                    width=_grid_plot_height_and_width,
                    height=_grid_plot_height_and_width,
                    output_backend="webgl",
               )
                # Add the histogram bars using quad glyphs
                p.quad(
                    source=self._hist_source[var_along_columns],
                    top="top",
                    bottom="bottom",
                    left="left",
                    right="right",
                    fill_color="#3288bd",
                    line_color="white",
                    alpha=0.7,
                )
                self._hist_figures[var_along_columns] = p

            if icolumn > irow: # only lower half and diagonal
                p = None
            else:
                # is it visible based on what the user has selected
                if self._sampled_variables_visibility[var_along_columns] and self._sampled_variables_visibility[var_along_rows]:
                    p.visible = True
                else:
                    p.visible = False

            plots_and_labels_in_grid.append(p)

    def _make_plot_labels(self, visible_variables, plots_and_labels_in_grid):
        # row labels
        for i, sampled_variable in enumerate(reversed(self._sampled_variables)):
            irow = self._num_sampled_variables - i - 1
            idx = irow * self._num_sampled_variables
            elided_variable_name_with_units = _elide_variable_name_with_units(sampled_variable, None)
            p = Div(
                text=f"<div style='text-align:center;font-size:12px;writing-mode:vertical-lr;transform:rotate(180deg); cursor:help;' title='{sampled_variable}'>{elided_variable_name_with_units}</div>",
                align="center",
            )
            self._row_labels[sampled_variable] = p
            p.visible = sampled_variable in visible_variables
            plots_and_labels_in_grid.insert(idx,p)

        # need to push one blank one for the bottom left corner in the grid
        p = Div(text=f"")
        plots_and_labels_in_grid.append(p)

        # column labels
        for sampled_variable in self._sampled_variables:
            units = self._case_tracker._get_units(sampled_variable)
            elided_variable_name_with_units = _elide_variable_name_with_units(sampled_variable, units)
            p = Div(
                text=f"<div style='text-align:center;font-size:14px; cursor:help;' title='{sampled_variable}'>{elided_variable_name_with_units}</div>",
                styles={"font-size": "12px", "text-align":"center"},
                align="center",
            )
            self._column_labels[sampled_variable] = p
            p.visible = sampled_variable in visible_variables
            plots_and_labels_in_grid.append(p)

    def _make_analysis_driver_box(self):
        analysis_progress_label = Div(
            text="Analysis Driver Progress",
            stylesheets = [style], css_classes = ['div_header']
        )

        # placeholder until we have data
        self._analysis_driver_progress_text_box = Div(
            text="""<div style="padding: 10px; border-radius: 5px;">
                    <p>Waiting for data...</p>
                    </div>""",
            width=_left_side_column_width,
            # height=100,
        )

        quit_button = self._make_quit_button()

        analysis_progress_box = Column(
            Row(analysis_progress_label, 
                Spacer(), 
                quit_button,
                # sizing_mode="stretch_width",
            ),
            self._analysis_driver_progress_text_box,
            width=_left_side_column_width,
            styles={
                "border-radius": "5px",
                "border": "5px solid black",
            },
        )
        return analysis_progress_box

    def _make_quit_button(self):
        quit_button = Button(label="Quit", button_type="danger")

        # Define callback function for the quit button
        def quit_app():
            raise KeyboardInterrupt("Quit button pressed")

        quit_button.on_click(quit_app)
        return quit_button

    def _make_overall_layout(self, analysis_progress_box, sampled_variables_box, color_bar, grid_of_plots):

        self._overall_layout = Row(
            Column(
                Spacer(height=20),
                analysis_progress_box,
                Spacer(height=20),
                sampled_variables_box,
                Spacer(height=20),
                sizing_mode="stretch_both",
            ),
            color_bar,
            Spacer(width=100, height=0), # move the column away from the color bar
            grid_of_plots,
            sizing_mode="stretch_height",
        )


def realtime_opt_plot(
    case_recorder_filename, callback_period, pid_of_calling_script, show
):
    """
    Visualize the objectives, desvars, and constraints during an optimization process.

    Parameters
    ----------
    case_recorder_filename : MetaModelStructuredComp or MetaModelUnStructuredComp
        The metamodel component.
    callback_period : float
        The time period between when the application calls the update method.
    pid_of_calling_script : int
        The process id of the calling optimization script, if called this way.
    show : bool
        If true, launch the browser display of the plot.
    """

    def _make_realtime_opt_plot_doc(doc):
        _RealTimeOptPlot(
            case_recorder_filename,
            callback_period,
            doc=doc,
            pid_of_calling_script=pid_of_calling_script,
        )

    _port_number = get_free_port()

    try:
        server = Server(
            {"/": Application(FunctionHandler(_make_realtime_opt_plot_doc))},
            port=_port_number,
            unused_session_lifetime_milliseconds=_unused_session_lifetime_milliseconds,
        )
        server.start()
        if show:
            server.io_loop.add_callback(server.show, "/")

        print(
            f"Real-time optimization plot server running on http://localhost:{_port_number}"
        )
        server.io_loop.start()
    except KeyboardInterrupt as e:
        print(
            f"Real-time optimization plot server stopped due to keyboard interrupt: {e}"
        )
    except Exception as e:
        print(f"Error starting real-time optimization plot server: {e}")
    finally:
        print("Stopping real-time optimization plot server")
        if "server" in globals():
            server.stop()