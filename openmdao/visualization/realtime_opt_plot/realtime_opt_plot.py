"""A real-time plot monitoring the optimization process as an OpenMDAO script runs."""

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
    from bokeh.models import (BasicTicker, ColumnDataSource, DataRange1d,
                            Grid, LassoSelectTool, LinearAxis, PanTool,
                            Plot, ResetTool, Scatter, WheelZoomTool)

    from bokeh.models import ColumnDataSource, LinearColorMapper, ColorBar

    from bokeh.io import show

    from bokeh.models import (
        ColumnDataSource,
        LinearAxis,
        Range1d,
        Toggle,
        Button,
        Column,
        Row,
        CustomJS,
        Div,
        ScrollBox,
        SingleIntervalTicker
    )


    from bokeh.layouts import gridplot
    from bokeh.models import (BasicTicker, ColumnDataSource, DataRange1d,
                            Grid, LassoSelectTool, LinearAxis, PanTool,
                            Plot, ResetTool, Scatter, WheelZoomTool)
    from bokeh.sampledata.penguins import data
    from bokeh.transform import factor_cmap

    from bokeh.models import ColumnDataSource, LinearColorMapper, ColorBar
    from bokeh.transform import transform


    from bokeh.models import (
        ColumnDataSource,
        LinearAxis,
        Range1d,
        Toggle,
        Button,
        Column,
        Row,
        CustomJS,
        Div,
        ScrollBox,
        SingleIntervalTicker
    )



    from bokeh.models.tools import (
        BoxZoomTool,
        ResetTool,
        HoverTool,
        PanTool,
        WheelZoomTool,
        SaveTool,
        ZoomInTool,
        ZoomOutTool,
    )
    from bokeh.plotting import figure
    from bokeh.server.server import Server
    from bokeh.application.application import Application
    from bokeh.application.handlers import FunctionHandler
    from bokeh.palettes import Category20, Colorblind
    bokeh_available = True
except ImportError:
    bokeh_available = False

import numpy as np

from openmdao.recorders.sqlite_reader import SqliteCaseReader
from openmdao.recorders.case import Case

try:
    from openmdao.utils.gui_testing_utils import _get_free_port
except ImportError:
    # If _get_free_port is unavailable, the default port will be used
    def _get_free_port():
        return 5000

# Constants
# the time between calls to the udpate method
# _time_between_callbacks_in_ms = 1000
_time_between_callbacks_in_ms = 100
# Number of milliseconds for unused session lifetime
_unused_session_lifetime_milliseconds = 1000 * 60 * 10
# color of the plot line for the objective function
_obj_color = "black"
# color of the buttons for variables not being shown
_non_active_plot_color = "black"
_plot_line_width = 3
# how transparent is the area part of the plot for desvars that are vectors
_varea_alpha = 0.3

# colors used for the plot lines and associated buttons and axes labels
# start with color-blind friendly colors and then use others if needed
if bokeh_available:
    _colorPalette = Colorblind[8] + Category20[20]

def _is_process_running(pid):
    if sys.platform == "win32":
        # PROCESS_QUERY_LIMITED_INFORMATION is available on Windows Vista and later.
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

        # Attempt to open the process.
        handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
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

    parser.add_argument('--pid', type=int, default=None,
                        help='Process ID of calling optimization script, '
                        'defaults to None if called by the user directly')
    parser.add_argument('--no-display', action='store_false', dest='show',
                        help="Do not launch browser showing plot. Used for CI testing")


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
        realtime_opt_plot(options.case_recorder_filename, _time_between_callbacks_in_ms,
                          options.pid, options.show)
    else:
        print("The bokeh library is not installed so the real-time optimizaton "
              "lot is not available. ")
        return

class _CaseRecorderTracker:
    """
    A class that is used to get information from a case recorder.

    These methods are not provided by the SqliteCaseReader class.
    """

    def __init__(self, case_recorder_filename):
        self._case_recorder_filename = case_recorder_filename
        self._cr = None
        self._initial_case = None  # need the initial case to get info about the variables
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
            cur.execute("SELECT * FROM driver_iterations WHERE "
                        "counter=:counter",
                        {"counter": counter})
            row = cur.fetchone()
        con.close()

        # use SqliteCaseReader code to get the data from this case
        if row:
            # TODO would be better to not have to open up the file each time
            self._open_case_recorder()
            var_info = self._cr.problem_metadata['variables']
            case = Case('driver', row, self._cr._prom2abs, self._cr._abs2prom, self._cr._abs2meta,
                        self._cr._conns, var_info, self._cr._format_version)

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

    def __init__(self, case_recorder_filename, callback_period, doc, pid_of_calling_script):
        """
        Construct and initialize _RealTimeOptPlot instance.
        """
        self._case_recorder_filename = case_recorder_filename
        self._case_tracker = _CaseRecorderTracker(case_recorder_filename)
        self._pid_of_calling_script = pid_of_calling_script

        self._sampled_variables = None
        self._prom_response = None

        self._num_samples_plotted = 0

        self._source = None
        self._lines = []
        # flag to prevent updating label with units each time we get new data
        self._labels_updated_with_units = False
        self._source_stream_dict = None

        self._hist_source = {}

        # used to keep track of the y min and max of the data so that
        # the axes ranges can be adjusted as data comes in
        self._y_min = defaultdict(lambda: float("inf"))
        self._y_max = defaultdict(lambda: float("-inf"))

        self._prom_response_min = float("inf")
        self._prom_response_max = float("-inf")

        self._start_time = time.time()

        self._hist_figures = {}

        def _update():
            # this is the main method of the class. It gets called periodically by Bokeh
            # It looks for new data and if found, updates the plot with the new data
            new_case = self._case_tracker._get_new_case()

            if new_case is None:
                if self._pid_of_calling_script is None or not _is_process_running(
                    self._pid_of_calling_script
                ):
                    # no more new data in the case recorder file and the
                    #   optimization script stopped running, so no possible way to
                    #   get new data.
                    # But just keep sending the last data point.
                    # This is a hack to force the plot to re-draw.
                    # Otherwise if the user clicks on the variable buttons, the
                    #   lines will not change color because of the set_value hack done to get
                    #   get around the bug in setting the line color from JavaScript
                    self._source.stream(self._source_stream_dict)
                return

            # See if Bokeh source object is defined yet. If not, set it up
            # since now we have data from the case recorder with info about the
            # variables to be plotted.
            if self._source is None:
                self._setup_data_source()
                self._setup_figure()

                # Check to make sure we have one and only one objective before going farther
                # obj_names = self._case_tracker._get_obj_names()
                # if len(obj_names) != 1:
                #     raise ValueError(
                #         f"Plot requires there to be one and only one objective \
                #             but {len(obj_names)} objectives found"
                #     )

                # quit_button = Button(label="Quit Application", button_type="danger")

                # # Define callback function for the quit button
                # def quit_app():
                #     raise KeyboardInterrupt("Quit button pressed")

                # # Attach the callback to the button
                # quit_button.on_click(quit_app)

                graph = Row(self.plot_figure, sizing_mode="stretch_both")
                doc.add_root(graph)
                # end of self._source is None - plotting is setup

            # Do the actual update of the plot including updating the plot range and adding the new
            # data to the Bokeh plot stream
            self._source_stream_dict = {}

            self._source_stream_dict[self._prom_response] = new_case.get_val(self._prom_response)[:1]

            for sampled_variable in self._sampled_variables:
                self._source_stream_dict[sampled_variable] = new_case.get_val(sampled_variable)[:1]

            self._source.stream(self._source_stream_dict)
            self._num_samples_plotted += 1

            self._prom_response_min = min(self._prom_response_min, new_case.get_val(self._prom_response)[:1][0])
            self._prom_response_max = max(self._prom_response_max, new_case.get_val(self._prom_response)[:1][0])

            self._color_mapper.low = self._prom_response_min
            self._color_mapper.high = self._prom_response_max

            # Get current date and time
            now = datetime.now()
            formatted_time = now.strftime("%H:%M:%S on %B %d, %Y")

            # Format it as requested

            current_time = time.time()
            elapsed_total = current_time - self._start_time
            elapsed_formatted = str(timedelta(seconds=int(elapsed_total))
)

            stats_text = f"""<div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px;">
                            <h3>Analysis Progress</h3>
                            <p>Number of samples: {self._num_samples_plotted}</p>
                            <p>Last updated: {formatted_time}</p>
                            <p>Elapsed time: {elapsed_formatted}</p>
                            </div>"""

            self._text_box.text = stats_text

            self._update_histograms()

            # end of _update method

        doc.add_periodic_callback(_update, callback_period)
        doc.title = "OpenMDAO Analysis Progress Plot"

    def _update_histograms(self):
        for sampled_variable in self._sampled_variables:
            x_data = self._source.data[sampled_variable]

            # Compute histogram with 30 bins
            # TODO make 30 a variable
            hist, edges = np.histogram(x_data, bins=30, range=(np.min(x_data), np.max(x_data)))

            self._hist_source[sampled_variable].data.update(
                {
                    'top':hist,
                    'left': edges[:-1],            # left edge of each bin
                    'right': edges[1:],            # right edge of each bin
                 }
                )

            self._hist_figures[sampled_variable].x_range = Range1d(np.min(x_data), np.max(x_data))

    def _setup_data_source(self):
        self._source_dict = {}

        # return prom
        outputs = self._case_tracker.get_case_reader().list_source_vars('driver')['outputs']

        # returns abs
        responses = list(self._case_tracker.get_case_reader().problem_metadata['responses'].keys())

        # convert to prom
        prom_responses = []
        for response in responses:
            if response in self._case_tracker.get_case_reader()._abs2prom['output']:
                prom_responses.append(self._case_tracker.get_case_reader()._abs2prom['output'][response])
            else:
                raise RuntimeError(f"No prom for abs variable {response}")

        # Look to see if there is an objective. If yes, that is the response to show
        # Case.get_objectives
        # cr = self._case_tracker.get_case_reader()
        # driver_cases = cr.list_cases('driver')

        driver_case = self._case_tracker._get_case_by_counter(1)
        objs = driver_case.get_objectives()
        # for now assume one response
        self._prom_response = prom_responses[0]
        self._source_dict[self._prom_response] = []



        self._sampled_variables = [varname for varname in outputs if varname not in prom_responses]
        for sampled_variable in self._sampled_variables:
            self._source_dict[sampled_variable] = []

        self._source = ColumnDataSource(self._source_dict)

    def _setup_figure(self):

        N = len(self._sampled_variables)

        xdrs = [DataRange1d(bounds=None) for _ in range(N)]
        ydrs = [DataRange1d(bounds=None) for _ in range(N)]

        # color_mapper = LinearColorMapper(palette=Greys256, low=min(source.data[prom_response]), high=max(source.data[prom_response]))

        from bokeh.palettes import Viridis256  # Viridis is colorblind-friendly

        # Create a color mapper using Viridis (colorblind-friendly)

        # color_mapper = LinearColorMapper(palette=Viridis256, low=min(self._source.data['obj']), high=max(self._source.data['obj']))

        self._color_mapper = LinearColorMapper(palette=Viridis256, low =0 , high=200)

        # For alpha values, we need to handle it differently
        # Instead of creating a LinearColorMapper for alpha, use a different approach:
        min_alpha = 0.2
        max_alpha = 0.9

        # # Calculate alpha values directly based on f
        # f_min = min(source.data[prom_response])
        # f_max = max(source.data[prom_response])
        # normalized_f = [(val - f_min) / (f_max - f_min) for val in source.data[prom_response]]
        # alpha_values = [min_alpha + norm_val * (max_alpha - min_alpha) for norm_val in normalized_f]

        # # Add alpha to the source
        # source.data['alpha'] = alpha_values

        plots = []

        # for i, (y, x) in enumerate(product(sampled_variables, reversed(sampled_variables))):
        for i, (y, x) in enumerate(product(self._sampled_variables, self._sampled_variables)):

            column = i % N
            row = i//N

            if x != y:

                x_units = self._case_tracker._get_units(x)
                y_units = self._case_tracker._get_units(y)


                # p = Plot(x_range=xdrs[i%N], y_range=ydrs[i//N],
                #         background_fill_color="#fafafa",
                #         border_fill_color="white", width=200, height=200, min_border=5)
                p = figure(x_range=xdrs[i%N], y_range=ydrs[i//N],
                        background_fill_color="#fafafa",
                        # border_fill_color="white", width=200, height=200, min_border=5)
                        border_fill_color="white", width=240, height=240)

                p.xaxis.axis_label = f"{x} ({x_units})"
                p.yaxis.axis_label = f"{y} ({y_units})"



                # p.xaxis.axis_label = x
                # p.yaxis.axis_label = y

                p.axis.visible = True

                # if i % N == 0:  # first column
                #     p.min_border_left = p.min_border + 4
                #     p.width += 40
                #     yaxis = LinearAxis(axis_label=y)
                #     yaxis.major_label_orientation = "vertical"
                #     p.add_layout(yaxis, "left")
                #     yticker = yaxis.ticker
                # else:
                #     yticker = BasicTicker()
                # p.add_layout(Grid(dimension=1, ticker=yticker))

                # if i >= N*(N-1):  # last row
                #     p.min_border_bottom = p.min_border + 40
                #     p.height += 40
                #     xaxis = LinearAxis(axis_label=x)
                #     p.add_layout(xaxis, "below")
                #     xticker = xaxis.ticker
                # else:
                #     xticker = BasicTicker()
                # p.add_layout(Grid(dimension=0, ticker=xticker))

                # scatter = Scatter(
                #     x=x,
                #     y=y,
                #     size=5,
                #     line_color=None,
                #     fill_color=transform(self._prom_response, color_mapper),  # This maps f to colors
                #     fill_alpha="alpha",
                # )
                # r = p.add_glyph(self._source, scatter)
                # p.x_range.renderers.append(r)
                # p.y_range.renderers.append(r)

                scatter = p.scatter(
                    x=x,
                        source=self._source,
                    y=y,
                    size=5,
                    line_color=None,
                    fill_color=transform(self._prom_response, self._color_mapper),  # This maps f to colors
                    # fill_alpha="alpha",
                )

            else:  # on the diagonal
                # Extract the x column data for the histogram
                x_data = self._source.data[x]

                # Compute histogram with 30 bins

                # DO I EVEN NEED THIS HERE? OR JUST in update?
                if x_data:
                    hist, edges = np.histogram(x_data, bins=30, range=(np.min(x_data), np.max(x_data)))
                else:
                    hist, edges = np.histogram(x_data, bins=30)

                # Create a new ColumnDataSource for the histogram data
                # hist_source = ColumnDataSource(data={
                self._hist_source[x] = ColumnDataSource(data={
                    'top': hist,                   # height of each bin
                    'bottom': np.zeros(len(hist)), # bottom of each bin starts at 0
                    'left': edges[:-1],            # left edge of each bin
                    'right': edges[1:],            # right edge of each bin
                })


                units = self._case_tracker._get_units(x)

                # Create the figure
                p = figure(
                    title=f"Histogram of {x} Values",
                        # For reasons TBD, the width and height in the Plot constructor
                        # and the width and height here mean different things.
                        # so need to add 40. TODO do this better
                        width=240, height=240,
                        # tools="pan,wheel_zoom,box_zoom,reset,save",
                        x_axis_label=f'{x} ({units})',
                        y_axis_label='Frequency')

                # Add the histogram bars using quad glyphs
                p.quad(source=self._hist_source[x],
                    top='top',
                    bottom='bottom',
                    left='left',
                    right='right',
                    fill_color="#3288bd",
                    line_color="white",
                    alpha=0.7)

                self._hist_figures[x] = p

            # suppress the diagonal
            # if (i%N) + (i//N) == N-1:
            # if x == y:
            if column > row:
                # r.visible = False
                # p.grid.grid_line_color = None
                p.visible = False

            # TODO needed?
            # p.add_tools(PanTool(), WheelZoomTool(), ResetTool(), LassoSelectTool())

            plots.append(p)

        gp = gridplot(plots, ncols=N, toolbar_location=None,
                    
                                #  tools="pan,wheel_zoom,box_zoom,reset,save",  # 'save' is included here
                                #  toolbar_location=None,  # 'save' is included here

                    
                    )

        # # Create a figure just for the color bar
        # color_bar_fig = figure(width=100,
        #                        height=gp.height,
        #                     toolbar_location=None,
        #                     min_border=10,
        #                     outline_line_color=None)

        # print(f"{plots[0].height=}")
        # p = figure(title="Simple Line Example", x_axis_label='x', y_axis_label='y', height=500, width=0)
        p = figure(height=2 * plots[0].height, width=0, toolbar_location=None)

        # Define two points (x0, y0) and (x1, y1)
        x = [0, 1]
        y = [0, 1]

        # Plot a line between the two points
        line = p.line(x, y,)

        p.grid.grid_line_color = None

        # Add the color bar to this figure
        color_bar = ColorBar(color_mapper=self._color_mapper, 
                            title=f"Response variable: '{self._prom_response}'",
                            border_line_color=None,
                            width=20,
                     label_standoff = 14,
                     title_text_font_size = '20px',
        ticker=BasicTicker(),  # This is the key part you're missing
                        location=(0,0)
                            )

        p.add_layout(color_bar, 'right')

        line.visible = False
        # Hide axes
        p.xaxis.visible = False
        p.yaxis.visible = False

        # Hide grid lines
        p.xgrid.visible = False
        p.ygrid.visible = False

        # Hide the title
        p.title.visible = False

        # Optional: remove the border/spines (outline)
        p.outline_line_alpha = 0
        # p.toolbar.logo = None

        # show number of samples plotted
        self._text_box = Div(
            text="""<div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px;">
                    <h3>Analysys Progress</h3>
                    <p>Waiting for data...</p>
                    </div>""",
            width=600,
            height=100,
        )

        # p.add_layout(self._text_box,'right')

        from bokeh.layouts import row, column, Spacer

        # final_layout = row(gp, p)

        # final_layout = row(gp, p, self._text_box)

        p.min_border_right = 100


        spacer2 = Spacer(width=200, height=300)  # Between plot 2 (with colorbar) and text box
        spacer3 = Spacer(width=100, height=20)
        spacer4 = Spacer(width=10, height=10)

        script_name = self._case_recorder_filename

        title_div = Div(
                    text=f"Analysis Progress for {script_name}",
                    styles={"font-size": "20px", "font-weight": "bold"},
                )

        quit_button = Button(label="Quit Application", button_type="danger")

        # Define callback function for the quit button
        def quit_app():
            raise KeyboardInterrupt("Quit button pressed")

        # Attach the callback to the button
        quit_button.on_click(quit_app)


        # Add the text box after the grid in a row
        # final_layout = row(gp, p, spacer2, self._text_box, sizing_mode='fixed')
        final_layout = row(column(title_div, gp), p, spacer2, 
                           column(self._text_box, spacer3, quit_button), sizing_mode='fixed')

        # col1 = column(gp)
        # col2 = column(p)
        # col3 = column(self._text_box)

        # final_layout = row(col1, col2, col3)


        # from bokeh.layouts import layout

        # final_layout = layout(
        #     [[gp, p, self._text_box]], spacing=25
        # )  # Add explicit spacing between elements

        self.plot_figure = final_layout

        from bokeh.plotting import save, output_file

        # output_file("sellar_analysis_driver_scatterplot_matrix.html")
        # save(final_layout)

        # show(final_layout)
        # show(gp)

        # # Make the figure and all the settings for it
        # self.plot_figure = figure(
        #     tools=[
        #         PanTool(),
        #         WheelZoomTool(),
        #         ZoomInTool(),
        #         ZoomOutTool(),
        #         BoxZoomTool(),
        #         ResetTool(),
        #         SaveTool(),
        #     ],
        #     width_policy="max",
        #     height_policy="max",
        #     sizing_mode="stretch_both",
        #     title=f"Optimization Progress Plot for: {self._case_recorder_filename}",
        #     active_drag=None,
        #     active_scroll="auto",
        #     active_tap=None,
        #     output_backend="webgl",
        # )
        # self.plot_figure.x_range.follow = "start"
        # self.plot_figure.title.text_font_size = "14px"
        # self.plot_figure.title.text_color = "black"
        # self.plot_figure.title.text_font = "arial"
        # self.plot_figure.title.align = "left"
        # self.plot_figure.title.standoff = 40  # Adds 40 pixels of space below the title

        # self.plot_figure.xaxis.axis_label = "Driver iterations"
        # self.plot_figure.xaxis.minor_tick_line_color = None
        # self.plot_figure.xaxis.ticker = SingleIntervalTicker(interval=1)

        # self.plot_figure.axis.axis_label_text_font_style = "bold"
        # self.plot_figure.axis.axis_label_text_font_size = "20pt"


def realtime_opt_plot(case_recorder_filename, callback_period, pid_of_calling_script, show):
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

    _port_number = _get_free_port()

    try:
        server = Server(
            {"/": Application(FunctionHandler(_make_realtime_opt_plot_doc))},
            port=_port_number,
            unused_session_lifetime_milliseconds=_unused_session_lifetime_milliseconds,
        )
        server.start()
        if show:
            server.io_loop.add_callback(server.show, "/")

        print(f"Real-time optimization plot server running on http://localhost:{_port_number}")
        server.io_loop.start()
    except KeyboardInterrupt as e:
        print(f"Real-time optimization plot server stopped due to keyboard interrupt: {e}")
    except Exception as e:
        print(f"Error starting real-time optimization plot server: {e}")
    finally:
        print("Stopping real-time optimization plot server")
        if "server" in globals():
            server.stop()
