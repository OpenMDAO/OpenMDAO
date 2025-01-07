""" A real-plot of the optimization process"""

from bokeh.models import ColumnDataSource, Legend, LegendItem, LinearAxis, Range1d, Toggle, Column, Row, CustomJS
from bokeh.plotting import curdoc, figure
from bokeh.server.server import Server
from tornado.ioloop import IOLoop
from bokeh.palettes import Category10, Category20, d3
from bokeh.layouts import row, column, Spacer

import numpy as np


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


def _make_legend_item(varname, color):
    toggle = Toggle(
    label=varname,
    active=False,
    # width=120,
    height=20,
    margin=(0, 0, 2, 0)
    )
    
    # Add custom CSS styles for both active and inactive states
    toggle.stylesheets = [
        f"""
            .bk-btn {{
                color: {color};
                border-color: {color};
                background-color: white;
                font-size: 12pt;
                display: flex;
                align-items: center; /* Vertical centering */
                justify-content: center; /* Horizontal centering */
                height: 12px; /* Example height, adjust as needed */
                border-width: 0px; /* Adjust to desired thickness */
                border-style: solid; /* Ensures a solid border */
            }}

            .bk-btn.bk-active {{
                color: white;
                border-color: {color};
                background-color: {color};
                font-size: 12pt;
                display: flex;
                align-items: center; /* Vertical centering */
                justify-content: center; /* Horizontal centering */
                height: 12px; /* Example height, adjust as needed */
                border-width: 0px; /* Adjust to desired thickness */
                border-style: solid; /* Ensures a solid border */
            }}

            .bk-btn:focus {{
                outline: none; /* Removes the default focus ring */
            }}
        """
    ]
    


    return toggle
    


class CaseTracker:
    def __init__(self, case_recorder_filename):
        self._case_ids_read = []
        self._case_recorder_filename = case_recorder_filename
        # self._cr = SqliteCaseReader(case_recorder_filename)
        self.source = None
        
        self._initial_cr_with_one_case = None

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

    def get_obj_names(self):
        if self._initial_cr_with_one_case is None:
            cr = SqliteCaseReader(self._case_recorder_filename)
            case_ids = cr.list_cases("driver", out_stream=None)
            if len(case_ids) > 0:
                self._initial_cr_with_one_case = cr
            else:
                return None
        case_ids = self._initial_cr_with_one_case.list_cases("driver", out_stream=None)
        driver_case = self._initial_cr_with_one_case.get_case(case_ids[0])
        obj_vars = driver_case.get_objectives()
        return obj_vars.keys()

    def get_desvar_names(self):
        if self._initial_cr_with_one_case is None:
            cr = SqliteCaseReader(self._case_recorder_filename)
            case_ids = cr.list_cases("driver", out_stream=None)
            if len(case_ids) > 0:
                self._initial_cr_with_one_case = cr
            else:
                return None
        case_ids = self._initial_cr_with_one_case.list_cases("driver", out_stream=None)
        driver_case = self._initial_cr_with_one_case.get_case(case_ids[0])
        design_vars = driver_case.get_design_vars()
        return design_vars.keys()

    def get_cons_names(self):
        if self._initial_cr_with_one_case is None:
            cr = SqliteCaseReader(self._case_recorder_filename)
            case_ids = cr.list_cases("driver", out_stream=None)
            if len(case_ids) > 0:
                self._initial_cr_with_one_case = cr
            else:
                return None
        case_ids = self._initial_cr_with_one_case.list_cases("driver", out_stream=None)
        driver_case = self._initial_cr_with_one_case.get_case(case_ids[0])
        cons = driver_case.get_constraints()
        return cons.keys()

    def get_obj_names(self):
        if self._initial_cr_with_one_case is None:
            cr = SqliteCaseReader(self._case_recorder_filename)
            case_ids = cr.list_cases("driver", out_stream=None)
            if len(case_ids) > 0:
                self._initial_cr_with_one_case = cr
            else:
                return None
        case_ids = self._initial_cr_with_one_case.list_cases("driver", out_stream=None)
        driver_case = self._initial_cr_with_one_case.get_case(case_ids[0])
        obj_vars = driver_case.get_objectives()
        return obj_vars.keys()
    
    def get_units(self, name):
        if self._initial_cr_with_one_case is None:
            cr = SqliteCaseReader(self._case_recorder_filename)
            case_ids = cr.list_cases("driver", out_stream=None)
            if len(case_ids) > 0:
                self._initial_cr_with_one_case = cr
            else:
                return None
        case_ids = self._initial_cr_with_one_case.list_cases("driver", out_stream=None)
        driver_case = self._initial_cr_with_one_case.get_case(case_ids[0])
        try:
            units = driver_case._get_units(name)
        except RuntimeError as err:
            if str(err).startswith("Can't get units for the promoted name"):
                return "Ambiguous"
            raise
        except KeyError as err:
            return "Unavailable"
            
        return units


class RealTimeOptPlot(object):
    def __init__(self, case_recorder_filename, doc):
        
        print("RealTimeOptPlot.__init__")
        
        self._source = None

        case_tracker = CaseTracker(case_recorder_filename)
       
       
        from bokeh.models.tools import BoxZoomTool, ResetTool, HoverTool, PanTool

        # Make the figure and all the settings for it
        p = figure(
            tools=[BoxZoomTool(),ResetTool()],
            # tooltips="Data point @x has the value @y",
            # tools="xpan,xwheel_zoom,xbox_zoom,reset",
                   width_policy="max" , height_policy="max",
                   title=f"Real-time Optimization Progress Plot for: {case_recorder_filename}",
                   output_backend="webgl",
        )
        p.add_tools(PanTool(dimensions="width"))
        # tooltips=[("Value", "@y")]
        
        
        # tooltips = [
        #     ("index", "$index"),
        #     ("(x,y)", "($x, $y)"),
        # #     ("radius", "@radius"),
        # #     ("fill color", "$color[hex, swatch]:fill_color"),
        # #     ("fill color", "$color[hex]:fill_color"),
        # #     ("fill color", "$color:fill_color"),
        # #     ("fill color", "$swatch:fill_color"),
        # #     ("foo", "@foo"),
        # #     ("bar", "@bar"),
        # ]
        
        
        
        # Create tooltips dynamically for all variables
        # tooltips = [("index", "$index")]
        # for name in  list(case_tracker.get_obj_names()) + list(case_tracker.get_cons_names()) + list(case_tracker.get_desvar_names()) :
        #     tooltips.append((f'{name}', f'@{name}{{0.2f}}'))


        # hover = HoverTool(
        #     # tooltips=tooltips,
        #     # formatters={
        #     #     '@date': 'datetime',  # use 'datetime' formatter for '@date' field
        #     # },
        #     # mode='vline'  # display tooltips for all points on a vertical line
        # )
        # p.add_tools(hover)


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
                    
                    # print("source is none and new data")
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
                        
                        
                    # print(f"{len(obj_names)=}")
                    # print(f"{len(desvar_names)=}")
                    # print(f"{len(con_names)=}")
                        
                    self._source = ColumnDataSource(source_dict)

                    # print(f"{self=}")
                    # print(f"{id(self)=}")
                    # print(f"set {self._source=}")

                    #### make the lines and legends
                    palette = Category20[20]
                    i_color = 0  # index of line across all variables: obj, desvars, cons
                    legend_items = []

                    toggles = []
                    
                    lines = []
                    
                    axes = []

                    print("make Objective legend items, line")
                    # Objective
                    legend_items.append(LegendItem(label="OBJECTIVE"))  # the only way to make a header in Legends
                    print(f"getting obj names")
                    obj_names = case_tracker.get_obj_names()
                    print(f"done getting obj names")
                    if len(obj_names) != 1:
                        raise ValueError(f"Plot assumes there is on objective but {len(obj_names)} found")
                    for i, obj_name in enumerate(obj_names):
                        
                        color = palette[i_color % 20]
                        units = case_tracker.get_units(obj_name)

                        toggle = _make_legend_item(f"{obj_name} ({units})", color)
                        toggles.append(toggle)


                        i_color += 1
                        obj_line = p.line(x="iteration", y=obj_name, line_width=3, source=self._source,
                                          color=color)
                        
                        lines.append(obj_line)
                        
                        
                        legend_items.append(LegendItem(label=f"{obj_name} ({units})", renderers=[obj_line]))
                        p.yaxis.axis_label = obj_name
                        # print(f"{i_color=}")
                        
                        
                        hover = HoverTool(renderers=[obj_line], 
                            tooltips=[('Iteration', '@iteration'), 
                             (obj_name, '@{%s}' % obj_name)],
                            mode='vline', visible=False)

                        # Add the hover tools to the plot
                        p.add_tools(hover)



                    # desvars
                    legend_items.append(LegendItem(label="DESIGN VARS"))  # the only way to make a header in Legends
                    desvar_names = case_tracker.get_desvar_names()
                    for i, desvar_name in enumerate(desvar_names):
                        color = palette[i_color % 20]
                        units = case_tracker.get_units(desvar_name)
                        
                        toggle = _make_legend_item(f"{desvar_name} ({units})", color)
                        toggles.append(toggle)

                        desvar_line = p.line(x="iteration", y=desvar_name, line_width=3, 
                                y_range_name=f"extra_y_{desvar_name}",
                                source=self._source,color=color, visible=False)
                        desvar_line.visible = False

                        lines.append(desvar_line)

                        hover = HoverTool(renderers=[desvar_line], 
                            tooltips=[('Iteration', '@iteration'), 
                             (desvar_name, '@{%s}' % desvar_name)],
                            mode='vline', visible=False)

                        # Add the hover tools to the plot
                        p.add_tools(hover)

                        legend_items.append(LegendItem(label=f"{desvar_name} ({units})", renderers=[desvar_line]))
                        extra_y_axis = LinearAxis(y_range_name=f"extra_y_{desvar_name}",
                                                axis_label=f"{desvar_name}",
                                                axis_label_text_color=color)

                        axes.append(extra_y_axis)

                        p.add_layout(extra_y_axis, 'right')
                        p.right[i_color-1].visible = False

                        
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

                        # p.add_layout(extra_y_axis, 'right')
                        i_color += 1

                    print("make cons legend items, line")

                    # cons
                    legend_items.append(LegendItem(label="CONSTRAINTS"))  # the only way to make a header in Legends
                    cons_names = case_tracker.get_cons_names()
                    for i, cons_name in enumerate(cons_names):
                        color = palette[i_color % 20]

                        units = case_tracker.get_units(cons_name)

                        toggle = _make_legend_item(f"{cons_name} ({units})", color)
                        toggles.append(toggle)

                        cons_line = p.line(x="iteration", y=cons_name, line_width=3, 
                            y_range_name=f"extra_y_{cons_name}",
                            source=self._source,color=color, visible=False)
                        # legend_items.append(LegendItem(label=f"{cons_name} ({units})", renderers=[cons_line]))

                        lines.append(cons_line)

                        extra_y_axis = LinearAxis(y_range_name=f"extra_y_{cons_name}",
                                                axis_label=f"{cons_name}",
                                                axis_label_text_color=color)

                        axes.append(extra_y_axis)
                        p.add_layout(extra_y_axis, 'right')
                        p.right[i_color-1].visible = False


                        
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

                        # p.add_layout(extra_y_axis, 'right')
                        i_color += 1


                    print("create actual Legend")
                    legend = Legend(items=legend_items, title="Variables")
                    
                    # p.add_layout(legend, "right")
                    
                    # num_lines = 50
                    # button_colors = [f'#{hash(str(i))& 0xFFFFFF:06x}' for i in range(num_lines)]

                    # # Create a toggle button styled with the line color
                    # for i in range(num_lines):
                    #     toggle = Toggle(
                    #         label=f"Line {i+1}",
                    #         active=False,
                    #         width=120,
                    #         height=15,
                    #         margin=(0, 0, 2, 0)
                    #     )
                        
                    #     # Add custom CSS styles for both active and inactive states
                    #     toggle.stylesheets = [
                    #         f"""
                    #             .bk-btn {{
                    #                 color: {button_colors[i]};
                    #                 border-color: {button_colors[i]};
                    #                 background-color: white;
                    #                 font-size: 8pt;
                    #                 display: flex;
                    #                 align-items: center; /* Vertical centering */
                    #                 justify-content: center; /* Horizontal centering */
                    #                 height: 12px; /* Example height, adjust as needed */
                    #                 border-width: 0px; /* Adjust to desired thickness */
                    #                 border-style: solid; /* Ensures a solid border */
                    #             }}

                    #             .bk-btn.bk-active {{
                    #                 color: white;
                    #                 border-color: {button_colors[i]};
                    #                 background-color: {button_colors[i]};
                    #                 font-size: 8pt;
                    #                 display: flex;
                    #                 align-items: center; /* Vertical centering */
                    #                 justify-content: center; /* Horizontal centering */
                    #                 height: 12px; /* Example height, adjust as needed */
                    #                 border-width: 0px; /* Adjust to desired thickness */
                    #                 border-style: solid; /* Ensures a solid border */
                    #             }}

                    #             .bk-btn:focus {{
                    #                 outline: none; /* Removes the default focus ring */
                    #             }}
                    #         """
                    #     ]
                        
                        # toggles.append(toggle)

                    # Create CustomJS callback for toggle buttons
                    callback = CustomJS(args=dict(lines=lines, axes=axes, toggles=toggles), code="""
                        // Get the toggle that triggered the callback
                        const toggle = cb_obj;
                        const index = toggles.indexOf(toggle);
                        
                        // Set line visibility
                        lines[index].visible = toggle.active;
                        
                        // Set axis visibility if it exists (all except first line)
                        if (index > 0 && index-1 < axes.length) {
                            axes[index-1].visible = toggle.active;
                        }
                    """)

                    # Add callback to all toggles
                    for toggle in toggles:
                        toggle.js_on_change('active', callback)

                    # Create a column of toggles with scrolling
                    toggle_column = Column(
                        children=toggles,
                        # width=150,
                        height=400,
                        sizing_mode="stretch_width",
                        styles={
                            'overflow-y': 'auto', 
                            'border': '1px solid #ddd',
                            'padding': '8px',
                            'background-color': '#f8f9fa'
                        }
                    )

                    
                    # p.add_layout(toggle_column, "right")
                    
                    graph = Row(p, toggle_column, sizing_mode='stretch_width')
                    
                    
            #                     layout="fit_data_stretch",
            # max_height=600,
            # sizing_mode='scale_both',

                    
                    doc.add_root(graph)

                    
                    p.legend.click_policy="hide"

                    print("end of source is none and new data")
                    
            if new_data is None:
                print("getting new cases")
                new_data = case_tracker.get_new_cases()
                print("done getting new cases")
            if new_data:
                print("new_data at the end")
                counter = new_data["counter"]
                source_stream_dict = {"iteration": [counter]}
                
                for obj_name, obj_value in new_data["objs"].items():

                    float_obj_value = 0. if obj_value is None or obj_value.size == 0 else np.linalg.norm(obj_value)

                    source_stream_dict[obj_name] = [float_obj_value]
                for desvar_name, desvar_value in new_data["desvars"].items():

                    float_desvar_value = 0. if desvar_value is None or desvar_value.size == 0 else np.linalg.norm(desvar_value)

                    source_stream_dict[desvar_name] = [float_desvar_value]

                    
                    self.y_min[desvar_name] = min(self.y_min[desvar_name], float_desvar_value)
                    self.y_max[desvar_name] = max(self.y_max[desvar_name], float_desvar_value)
                    if self.y_min[desvar_name] == self.y_max[desvar_name]:
                        self.y_min[desvar_name]  = self.y_min[desvar_name] - 1
                        self.y_max[desvar_name]  = self.y_max[desvar_name]  + 1
                    p.extra_y_ranges[f"extra_y_{desvar_name}"] = Range1d(
                            self.y_min[desvar_name], self.y_max[desvar_name])

                for cons_name, cons_value in new_data["cons"].items():

                    float_cons_value = 0. if cons_value is None or cons_value.size == 0 else np.linalg.norm(cons_value)

                    source_stream_dict[cons_name] = [float_cons_value]


                    self.y_min[cons_name] = min(self.y_min[cons_name], float_cons_value)
                    self.y_max[cons_name] = max(self.y_max[cons_name], float_cons_value)
                    if self.y_min[cons_name] == self.y_max[cons_name]:
                        self.y_min[cons_name]  = self.y_min[cons_name] - 1
                        self.y_max[cons_name]  = self.y_max[cons_name]  + 1
                    p.extra_y_ranges[f"extra_y_{cons_name}"] = Range1d(
                            self.y_min[cons_name], self.y_max[cons_name])
                self._source.stream(source_stream_dict)
                print("done new_data at the end")

        print("adding root")
        
        
        
        # Add custom CSS to make the legend scrollable
        custom_css = """
        <style>
        .bk-legend {
            max-height: 200px;
            overflow-y: auto;
            border: 1px solid black;
        }
        </style>
        """

        from bokeh.resources import INLINE
        # Inject the custom CSS into the Bokeh document
        curdoc().template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <title>Scrollable Legend</title>
            {custom_css}
            {{ bokeh_css }}
            {{ bokeh_js }}
        </head>
        <body>
            {{ plot_div | safe }}
            {{ plot_script | safe }}
        </body>
        </html>
        """
        
        
        
        
        
        
        
        # doc.add_root(p)
        # doc.add_root(graph)
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
        # import bokeh.settings.settings
        # settings.log_level = 'debug'
        # from bokeh.util.info import print_info
        # print_info()
        # from bokeh.util.logconfig import bokeh_logger as log
        # import logging
        # from bokeh.util.logconfig import basicConfig
        # basicConfig(level=logging.TRACE)

        # log.info(" -- info INIT TEXT")
        # log.error(" -- error INIT TEXT")

        server = Server({'/': Application(FunctionHandler(_make_realtime_opt_plot_doc))}, port=port_number, 
                        unused_session_lifetime_milliseconds=1000*60*10,
                        )
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
        if 'server' in globals():
            server.stop()
