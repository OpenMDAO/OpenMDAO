""" A real-plot of the optimization process"""

from bokeh.models import ColumnDataSource, Legend, LegendItem, LinearAxis, Range1d
from bokeh.plotting import curdoc, figure
from bokeh.server.server import Server
from tornado.ioloop import IOLoop
from bokeh.palettes import Category10, Category20, d3
from bokeh.layouts import row, column, Spacer
from openmdao.recorders.sqlite_reader import SqliteCaseReader

try:
    from openmdao.utils.gui_testing_utils import get_free_port
except:
    # If get_free_port is unavailable, the default port will be used
    def get_free_port():
        return 5000

def _realtime_opt_plot_setup_parser(parser):
    """
    Set up the realtime plot subparser for the 'openmdao realtime_opt_plot' command.

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

def _realtime_opt_plot_cmd(options, user_args):
    """
    Run the realtime_opt_plot command.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.
    user_args : list of str
        Args to be passed to the user script.
    """

    realtime_opt_plot(
        options.case_recorder_filename,
    )


class CaseTracker:
    def __init__(self, case_recorder_filename):
        self._case_ids_read = []
        self._case_recorder_filename = case_recorder_filename
        # self._cr = SqliteCaseReader(case_recorder_filename)
        self.source = None

    def get_new_cases(self):
        # need to read this each time since the constructor does all of the actual reading
        # TODO - add code SqliteCaseReader for reading real-time data
        # cr = SqliteCaseReader("create_cr_files_out/driver_history.db")
        self._cr = SqliteCaseReader(self._case_recorder_filename)
        case_ids = self._cr.list_cases("driver", out_stream=None)
        new_case_ids = [
            case_id for case_id in case_ids if case_id not in set(self._case_ids_read)
        ]
        if new_case_ids:
            # just get the first one
            case_id = new_case_ids[0]
            driver_case = self._cr.get_case(case_id)
            objs = driver_case.get_objectives()
            design_vars = driver_case.get_design_vars()
            constraints = driver_case.get_constraints()
            
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

            self._case_ids_read.append(case_id)  # remember that this one has been plotted

            return new_data
        return None

    def get_desvar_names(self):
        self._cr = SqliteCaseReader(self._case_recorder_filename)
        case_ids = self._cr.list_cases("driver", out_stream=None)
        driver_case = self._cr.get_case(case_ids[0])
        design_vars = driver_case.get_design_vars()
        return design_vars.keys()

    def get_cons_names(self):
        self._cr = SqliteCaseReader(self._case_recorder_filename)
        case_ids = self._cr.list_cases("driver", out_stream=None)
        driver_case = self._cr.get_case(case_ids[0])
        cons = driver_case.get_constraints()
        return cons.keys()

    def get_obj_names(self):
        self._cr = SqliteCaseReader(self._case_recorder_filename)
        case_ids = self._cr.list_cases("driver", out_stream=None)
        driver_case = self._cr.get_case(case_ids[0])
        obj_vars = driver_case.get_objectives()
        return obj_vars.keys()
    
    def get_units(self, name):
        self._cr = SqliteCaseReader(self._case_recorder_filename)
        case_ids = self._cr.list_cases("driver", out_stream=None)
        driver_case = self._cr.get_case(case_ids[0])
        try:
            units = driver_case._get_units(name)
        except RuntimeError as err:
            if str(err).startswith("Can't get units for the promoted name"):
                return "Ambiguous"
            raise
            
        return units


class RealTimeOptPlot(object):
    def __init__(self, case_recorder_filename, doc):
        
        self._source = None

        case_tracker = CaseTracker(case_recorder_filename)
       
        # Make the figure and all the settings for it
        p = figure(tools="xpan,xwheel_zoom,xbox_zoom,reset",
                   width_policy="max" , height_policy="max",
                   title="Real-time Optimization Progress Plot",
        )

        # Add an invisible line renderer. To avoid this warning message
        #     (MISSING_RENDERERS): Plot has no renderers
        p.line([], [], line_alpha=0)

        p.x_range.follow = "start"
        p.title.text_font_size = '25px'
        p.title.text_color = "black"
        p.title.text_font = "arial"
        p.title.align = "center"
        p.title.background_fill_color = "#cccccc"
        p.xaxis.axis_label = "Driver iterations"
        p.yaxis.axis_label = "Model variables"
        p.xaxis.minor_tick_line_color = None
        p.axis.axis_label_text_font_style = 'bold'
        p.axis.axis_label_text_font_size = '20pt'
        p.xgrid.band_hatch_pattern = "/"
        p.xgrid.band_hatch_alpha = 0.6
        p.xgrid.band_hatch_color = "lightgrey"
        p.xgrid.band_hatch_weight = 0.5
        p.xgrid.band_hatch_scale = 10
        
        from collections import defaultdict
        self.y_min = defaultdict(lambda: float("inf"))  # update this as new data comes in
        self.y_max = defaultdict(lambda: float("-inf"))  # update this as new data comes in

        def update():
            # See if source is defined yet. If not, see if we have any data
            #   in the case file yet. If there is data, create the
            #   source object and add the lines to the figure

            new_data = None
            
            if self._source is None:
                new_data = case_tracker.get_new_cases()
                if new_data:
                    ####  make the source dict
                    source_dict = { 'iteration': []}
                    
                    # Obj
                    obj_names = case_tracker.get_obj_names()
                    for obj_name in obj_names:
                        source_dict[obj_name] = []
                        
                    # Desvars
                    desvar_names = case_tracker.get_desvar_names()
                    for desvar_name in desvar_names:
                        source_dict[desvar_name] = []
                        
                    # Cons
                    con_names = case_tracker.get_cons_names()
                    for con_name in con_names:
                        source_dict[con_name] = []
                        
                    self._source = ColumnDataSource(source_dict)

                    #### make the lines and legends
                    palette = Category20[20]
                    i_color = 0  # index of line across all variables: obj, desvars, cons
                    legend_items = []

                    # Objective
                    legend_items.append(LegendItem(label="OBJECTIVE"))  # the only way to make a header in Legends
                    obj_names = case_tracker.get_obj_names()
                    if len(obj_names) != 1:
                        raise ValueError(f"Plot assumes there is on objective but {len(obj_names)} found")
                    for i, obj_name in enumerate(obj_names):
                        color = palette[i_color % 20]
                        i_color += 1
                        obj_line = p.line(x="iteration", y=obj_name, line_width=3, source=self._source,
                                          color=color)
                        units = case_tracker.get_units(obj_name)
                        legend_items.append(LegendItem(label=f"{obj_name} ({units})", renderers=[obj_line]))
                        p.yaxis.axis_label = obj_name


                    # desvars
                    legend_items.append(LegendItem(label="DESIGN VARS"))  # the only way to make a header in Legends
                    desvar_names = case_tracker.get_desvar_names()
                    for i, desvar_name in enumerate(desvar_names):
                        color = palette[i_color % 20]
                        i_color += 1
                        desvar_line = p.line(x="iteration", y=desvar_name, line_width=3, 
                                y_range_name=f"extra_y_{desvar_name}",
                                source=self._source,color=color)
                        units = case_tracker.get_units(desvar_name)
                        legend_items.append(LegendItem(label=f"{desvar_name} ({units})", renderers=[desvar_line]))

                        extra_y_axis = LinearAxis(y_range_name=f"extra_y_{desvar_name}",
                                                axis_label=f"{desvar_name}",
                                                axis_label_text_color=color)
                        
                        # set the range
                        y_min = -20
                        y_max = -20
                        # if the range is zero, the axis will not be displayed. Plus need some range to make it
                        #    look good. Some other code seems to do +- 1 for the range in this case.
                        if y_min == y_max:
                            y_min = y_min - 1
                            y_max = y_max + 1
                        p.extra_y_ranges[f"extra_y_{desvar_name}"] = Range1d(
                            y_min, y_max)

                        p.add_layout(extra_y_axis, 'right')


                    # cons
                    legend_items.append(LegendItem(label="CONSTRAINTS"))  # the only way to make a header in Legends
                    cons_names = case_tracker.get_cons_names()
                    for i, cons_name in enumerate(cons_names):
                        color = palette[i_color % 20]
                        i_color += 1
                        cons_line = p.line(x="iteration", y=cons_name, line_width=3, 
                            y_range_name=f"extra_y_{cons_name}",
                            source=self._source,color=color)
                        units = case_tracker.get_units(cons_name)
                        legend_items.append(LegendItem(label=f"{cons_name} ({units})", renderers=[cons_line]))


                        extra_y_axis = LinearAxis(y_range_name=f"extra_y_{cons_name}",
                                                axis_label=f"{cons_name}",
                                                axis_label_text_color=color)
                        
                        # set the range
                        y_min = -100
                        y_max = 100
                        # if the range is zero, the axis will not be displayed. Plus need some range to make it
                        #    look good. Some other code seems to do +- 1 for the range in this case.
                        if y_min == y_max:
                            y_min = y_min - 1
                            y_max = y_max + 1
                        p.extra_y_ranges[f"extra_y_{cons_name}"] = Range1d(
                            y_min, y_max)

                        p.add_layout(extra_y_axis, 'right')



                    legend = Legend(items=legend_items, title="Variables")
                    p.add_layout(legend, "right")

            if new_data is None:
                new_data = case_tracker.get_new_cases()
            if new_data:
                counter = new_data["counter"]
                source_stream_dict = {"iteration": [counter]}
                
                for obj_name, obj_value in new_data["objs"].items():
                    source_stream_dict[obj_name] = obj_value
                for desvar_name, desvar_value in new_data["desvars"].items():
                    source_stream_dict[desvar_name] = desvar_value

                    self.y_min[desvar_name] = min(self.y_min[desvar_name], float(desvar_value))
                    self.y_max[desvar_name] = max(self.y_max[desvar_name], float(desvar_value))
                    if self.y_min[desvar_name] == self.y_max[desvar_name]:
                        self.y_min[desvar_name]  = self.y_min[desvar_name] - 1
                        self.y_max[desvar_name]  = self.y_max[desvar_name]  + 1
                    p.extra_y_ranges[f"extra_y_{desvar_name}"] = Range1d(
                            self.y_min[desvar_name], self.y_max[desvar_name])

                for cons_name, cons_value in new_data["cons"].items():
                    source_stream_dict[cons_name] = cons_value
                    self.y_min[cons_name] = min(self.y_min[cons_name], float(cons_value))
                    self.y_max[cons_name] = max(self.y_max[cons_name], float(cons_value))
                    if self.y_min[cons_name] == self.y_max[cons_name]:
                        self.y_min[cons_name]  = self.y_min[cons_name] - 1
                        self.y_max[cons_name]  = self.y_max[cons_name]  + 1
                    p.extra_y_ranges[f"extra_y_{cons_name}"] = Range1d(
                            self.y_min[cons_name], self.y_max[cons_name])
                self._source.stream(source_stream_dict)

        doc.add_root(p)
        doc.add_periodic_callback(update, 50)
        doc.title = "OpenMDAO Optimization"

    
def realtime_opt_plot(case_recorder_filename):
    """
    Visualize a ??.

    Parameters
    ----------
    case_recorder_filename : MetaModelStructuredComp or MetaModelUnStructuredComp
        The metamodel component.
    """
    from bokeh.application.application import Application
    from bokeh.application.handlers import FunctionHandler

    def _make_realtime_opt_plot_doc(doc):
        RealTimeOptPlot(case_recorder_filename, doc=doc)

    port_number = get_free_port()

    try:
        server = Server({'/': Application(FunctionHandler(_make_realtime_opt_plot_doc))}, port=port_number)
        server.start()
        server.io_loop.add_callback(server.show, "/")
        
        print(f"Bokeh server running on http://localhost:{port_number}")
        server.io_loop.start()
    except KeyboardInterrupt as e:
        print(f"Server stopped due to keyboard interrupt")
    except Exception as e:
        print(f"Error starting Bokeh server: {e}")
    finally:
        print("Stopping server")
        server.stop()
