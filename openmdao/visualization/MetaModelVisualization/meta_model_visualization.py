# Import 

# Bokeh Imports
from bokeh.io import output_file, show, curdoc
from bokeh.layouts import row, column
from bokeh.plotting import figure
from bokeh.layouts import gridplot
from bokeh.models import Axis, Slider, ColumnDataSource, ContinuousColorMapper, ColorBar, FixedTicker, BasicTicker, LinearColorMapper, Range1d
from bokeh.models.renderers import GlyphRenderer
from bokeh.models.widgets import TextInput, Select
from openmdao.devtools.debug import profiling


# Misc Imports
import numpy as np
import openmdao.api as om
import math
from scipy.spatial import cKDTree
from collections import OrderedDict


class UnstructuredMetaModelVisualization(object):
    
    def __init__(self, prob, surrogate_comp, resolution=50):

        self.prob = prob
        self.surrogate_comp = surrogate_comp
        self.n = resolution
        
        # Create list of inputs
        self.input_list = [i[0] for i in self.surrogate_comp._surrogate_input_names]
        if len(self.input_list) < 2:
            raise ValueError('Must have more than one input value')
        
        self.output_list = [i[0] for i in self.surrogate_comp._surrogate_output_names]

        # Pair input list names with their respective data
        self.input_data = {}
        for title in self.input_list:
            try:
                self.input_data[title] = {i for i in self.surrogate_comp.options[str('train:' + title)]}
            except:
                msg = "No training data present for one or more parameters"
                raise TypeError(msg)

        # Setup dropdown menus for x/y inputs and the output value
        self.x_input = Select(title="X Input:", value=[x for x in self.input_list][0], 
        options=[x for x in self.input_list])
        self.x_input.on_change('value', self.x_input_update)
        
        self.y_input = Select(title="Y Input:", value=[x for x in self.input_list][1], 
        options=[x for x in self.input_list])
        self.y_input.on_change('value', self.y_input_update)

        self.output_value = Select(title="Output:", value=[x for x in self.output_list][0], 
        options=[x for x in self.output_list])
        self.output_value.on_change('value', self.output_value_update)

        # Create sliders in a loop
        self.slider_dict = {}
        self.input_data_dict = OrderedDict()
        for title, values in self.input_data.items():
            slider_spacing = np.linspace(min(values), max(values), self.n)
            self.input_data_dict[title] = slider_spacing
            slider_step = slider_spacing[1] - slider_spacing[0] # Calculates the distance between slider ticks
            self.slider_dict[title] = Slider(start=min(values), end=max(values), value=min(values), step=slider_step, title=str(title))        

        # Match the slider dictionary key value pairs with an on change event handler to call an update function later
        for name, slider_object in self.slider_dict.items():
            if name == self.x_input.value:
                self.x_input_slider = slider_object
                self.x_input_slider.on_change('value', self.scatter_plots_update)
            elif name == self.y_input.value:
                self.y_input_slider = slider_object
                self.y_input_slider.on_change('value', self.scatter_plots_update)
            else:
                setattr(self, name, slider_object)
                obj = getattr(self, name)
                obj.on_change('value', self.update)

        # Length of inputs and outputs 
        self.nx = len(self.input_list)
        self.ny = len(self.output_list)

        # Positional indicies
        self.x_index = 0
        self.y_index = 1
        self.output_variable = self.output_list.index(self.output_value.value)

        # Most data sources are filled with initial values
        self.slider_source = ColumnDataSource(data=self.input_data_dict)
        self.bot_plot_source = ColumnDataSource(data=dict(bot_slice_x=np.repeat(0,self.n), bot_slice_y=np.repeat(0,self.n)))
        self.right_plot_source = ColumnDataSource(data=dict(left_slice_x=np.repeat(0,self.n), left_slice_y=np.repeat(0,self.n))) 
        self.source = ColumnDataSource(data=dict(x=np.repeat(0,self.n), y=np.repeat(0,self.n)))

        # Text input to change the distance of reach when searching for nearest data points
        self.scatter_distance = TextInput(value="0.1", title="Scatter Distance")
        self.scatter_distance.on_change('value', self.scatter_input)
        self.dist_range = float(self.scatter_distance.value)
        
        # Grouping all of the sliders and dropdowns into one column
        sliders = [i for i in self.slider_dict.values()]
        self.sliders_and_selects = row(
            column(*sliders, self.x_input,
            self.y_input, self.output_value, self.scatter_distance)
        )

        # Layout creation
        self.layout = row(self.contour_data(), self.right_plot(), self.sliders_and_selects)
        self.layout2 = row(self.bot_plot())
        curdoc().add_root(self.layout)
        curdoc().add_root(self.layout2)
        curdoc().title = 'MultiView'

    def make_predictions(self, data):
        """ Runs the data parameter through the surrogate model which is given in prob

        Parameters:
        data (dict): Dictionary containing Ordered Dict of training points

        Returns:
        array: np.stack of predicted points

        """
        
        outputs = {i : [] for i in self.output_list}
        print("Making Predictions")

        # Parse dict into shape [n**2, number of inputs] list
        inputs = np.empty([self.n**2, self.nx])
        for idx, values in enumerate(data.values()):
            inputs[:, idx] = values.flatten()

        # Pair data points with their respective prob name. Loop to make predictions
        for idx, tup in enumerate(inputs):
            for name, val in zip(data.keys(), tup):
                self.prob[self.surrogate_comp.name + '.' + name] = val
            self.prob.run_model()
            for i in self.output_list:
                outputs[i].append(float(self.prob[self.surrogate_comp.name + '.' + i]))

        return self.stack_outputs(outputs)

    def contour_data(self):
        """ Creates a contour plot

        Parameters:
        None

        Returns:
        Bokeh Image Plot

        """

        n = self.n
        xe = np.zeros((n, n, self.nx))
        ye = np.zeros((n, n, self.ny))

        # Query the slider dictionary, append the name and current value to the ordered dictionary
        self.slider_value_and_name = OrderedDict()
        for title, slider_params in self.slider_dict.items():
            self.slider_value_and_name[title] = slider_params.value
        
        # Cast the current values of the slider_value_and_name dictionary values to a list 
        self.input_point_list = list(self.slider_value_and_name.values())
        for ix in range(self.nx):
            xe[:, :, ix] = self.input_point_list[ix]

        # Search the input_data_dict to match the names with the x/y dropdown menus. Then set x/y linspaces 
        # to the values for the meshgrid which follows
        for title, values in self.input_data_dict.items():
            if title == self.x_input.value:
                xlins = values
                dw = max(values)
            if title == self.y_input.value:
                ylins = values
                dh = max(values)

        # Create a mesh grid and then append that data to 'xe' in the respective columns 
        X, Y = np.meshgrid(xlins, ylins)
        xe[:, :, self.x_index] = X
        xe[:, :, self.y_index] = Y

        # This block places the x and y inputs first and then appends any other values to the list the first
        # two points 
        pred_dict = {}
        self.input_list = [self.x_input.value, self.y_input.value]
        for title in self.slider_value_and_name.keys():
            if title == self.x_input.value or title == self.y_input.value:
                pass
            else:
                self.input_list.append(title)
        
        # Append the key (input_list) and the values copied from xe to pred_dict where it is then ordered
        # in pred_dict_ordered.
        for idx, title in enumerate(self.slider_value_and_name.keys()):
            pred_dict.update({title: xe[:, :, idx]})
        pred_dict_ordered = OrderedDict((k, pred_dict[k]) for k in self.input_list)

        # Pass the dict to make predictions and then reshape the output to (n, n, number of outputs)
        ye[:, :, :] = self.make_predictions(pred_dict_ordered).reshape((n, n, self.ny))
        Z = ye[:, :, self.output_variable]
        Z = Z.reshape(n, n)
        self.Z = Z

        self.source.add(Z, 'z')

        # Color bar formatting
        color_mapper =  LinearColorMapper(palette="Viridis11", low=np.amin(Z), high=np.amax(Z))
        color_bar = ColorBar(color_mapper=color_mapper, ticker=BasicTicker(), label_standoff=12, location=(0,0))

        # Contour Plot
        self.contour_plot = figure(tooltips=[(self.x_input.value, "$x"), (self.y_input.value, "$y"), (self.output_value.value, "@image")], tools="pan")
        self.contour_plot.x_range.range_padding = 0
        self.contour_plot.y_range.range_padding = 0
        self.contour_plot.plot_width = 600
        self.contour_plot.plot_height = 500
        self.contour_plot.xaxis.axis_label = self.x_input.value
        self.contour_plot.yaxis.axis_label = self.y_input.value
        self.contour_plot.min_border_left = 0
        self.contour_plot.add_layout(color_bar, 'right')
        self.contour_plot.x_range = Range1d(min(xlins), max(xlins))
        self.contour_plot.y_range = Range1d(min(ylins), max(ylins))

        self.contour_plot.image(image=[self.source.data['z']], x=min(xlins), y=min(ylins), dh=dh, dw=dw, palette="Viridis11")

        # Adding training data points overlay to contour plot
        data = self.training_points()
        if len(data):
            data = np.array(data)
            self.contour_plot.circle(x=data[:, 0], y=data[:,1], size=5, color='white', alpha=0.50)

        return self.contour_plot        

        

    def right_plot(self):

        """ Creates a the right side subplot to view the projected slice

        Parameters:
        None

        Returns:
        Bokeh figure

        """
        
        # Sets data for x/y inputs
        for title in self.input_data_dict.keys():
            if title == self.x_input.value:
                x_value = self.x_input_slider.value
                # Rounds the x_data to match the input_data_dict value 
                mach_index = np.where(np.around(self.input_data_dict[title], 5) == np.around(x_value, 5))[0]
            elif title == self.y_input.value:
                y_data = self.input_data_dict[title]

        # Make slice in Z data at the point calculated before and add it to the data source 
        z_data = self.Z[:, mach_index].flatten()
        self.source.add(z_data, 'left_slice')

        x = self.source.data['left_slice']
        y = self.slider_source.data[self.y_input.value]

        # Create and format figure 
        s1 = figure(plot_width=200, plot_height=500, x_range=(min(x), max(x)), y_range=(min(y_data),max(y_data)), title="{} vs {}".format(self.y_input.value, self.output_value.value), tools="")
        s1.xaxis.axis_label = self.output_value.value
        s1.yaxis.axis_label = self.y_input.value
        s1.line(x, y)

        # Determine distance and alpha opacity of training points
        data = self.training_points()
        vert_color = np.zeros((len(data), 1))
        for i,info in enumerate(data):
            alpha = np.abs(info[0] - x_value) / self.limit_range[self.x_index]
            if alpha < self.dist_range:
                vert_color[i, -1] = (1 - alpha / self.dist_range) * info[-1]

        color = np.column_stack((data[:,-4:-1] - 1, vert_color))
        alphas = [0 if math.isnan(x) else x for x in color[:, 3]]
        s1.scatter(x=data[:, 3], y=data[:, 1], line_color=None, fill_color='#000000', fill_alpha=alphas)

        # Set the right_plot data source to new values
        self.right_plot_source.data = dict(left_slice_x=np.repeat(x_value, self.n), left_slice_y=y_data, 
        x1=np.array([x+self.dist_range for x in np.repeat(x_value, self.n)]), x2=np.array([x-self.dist_range for x in np.repeat(x_value, self.n)]))

        self.contour_plot.line('left_slice_x', 'left_slice_y', source=self.right_plot_source, color='black', line_width=2)
        
        return s1

    def bot_plot(self):

        """ Creates a the bottom subplot to view the projected slice

        Parameters:
        None

        Returns:
        Bokeh figure

        """

        for title in self.input_data_dict.keys():
            if title == self.x_input.value:
                self.x_data = self.input_data_dict[title]
                
            elif title == self.y_input.value:
                self.y_value = self.y_input_slider.value
                alt_index = np.where(np.around(self.input_data_dict[title], 5) == np.around(self.y_value, 5))[0]
        
        z_data = self.Z[alt_index].flatten()
        self.source.add(z_data, 'bot_slice')

        x = self.slider_source.data[self.x_input.value]
        y = self.source.data['bot_slice']

        s2 = figure(plot_width=550, plot_height=200, x_range=(min(self.x_data),max(self.x_data)), y_range=(min(y), max(y)), 
        title="{} vs {}".format(self.x_input.value, self.output_value.value), tools="")
        s2.xaxis.axis_label = self.x_input.value
        s2.yaxis.axis_label = self.output_value.value
        s2.line(x, y)

        data = self.training_points()
        horiz_color = np.zeros((len(data), 1))
        for i,info in enumerate(data):
            alpha = np.abs(info[1] - self.y_value) / self.limit_range[self.y_index]
            if alpha < self.dist_range:
                horiz_color[i, -1] = (1 - alpha / self.dist_range) * info[-1]
        
        color = np.column_stack((data[:,-4:-1] - 1, horiz_color))
        alphas = [0 if math.isnan(x) else x for x in color[:, 3]]
        s2.scatter(x=data[:, 0], y=data[:, 3], line_color=None, fill_color='#000000', fill_alpha=alphas)

        self.bot_plot_source.data = dict(bot_slice_x=self.x_data, bot_slice_y=np.repeat(self.y_value, self.n))
        self.contour_plot.line('bot_slice_x', 'bot_slice_y', source=self.bot_plot_source, color='black', line_width=2)

        return s2

    def update_all_plots(self):
        self.layout.children[0] = self.contour_data()
        self.layout.children[1] = self.right_plot()
        self.layout2.children[0] = self.bot_plot()

    def update_subplots(self):
        self.layout.children[1] = self.right_plot()
        self.layout2.children[0] = self.bot_plot()


    # Event handler functions
    def update(self, attr, old, new):
        self.update_all_plots()

    def scatter_plots_update(self, attr, old, new):
        self.update_subplots()

    def scatter_input(self, attr, old, new):
        self.dist_range = float(new)
        self.update_all_plots()

    def input_dropdown_checks(self,x,y):
        # Checks to see if x and y inputs are equal to each other
        if x == y:
            return False
        else:
            return True
    
    def x_input_update(self, attr, old, new):
        if self.input_dropdown_checks(new, self.y_input.value) == False:
            raise ValueError("Inputs should not equal each other")
        else:
            self.update_all_plots()

    def y_input_update(self, attr, old, new):
        if self.input_dropdown_checks(self.x_input.value, new) == False:
            raise ValueError("Inputs should not equal each other")
        else: 
            self.update_all_plots()

    def output_value_update(self, attr, old, new):
        self.output_variable = self.output_list.index(new)
        self.update_all_plots()

    def training_points(self):
        """ Calculates the training points and returns and array containing the position and alpha opacity
        of the training points nearest to the surrogate line.

        Parameters:
        None

        Returns:
        Array: The array of training points and their alpha opacity with respect to the surrogate line

        """

        # xt contains
        # [x1, x2, x3, x4]
        xt = self.surrogate_comp._training_input # Input Data
        yt = np.squeeze(self.stack_outputs(self.surrogate_comp._training_output), axis=1) # Output Data
        output_variable = self.output_list.index(self.output_value.value)
        data = np.zeros((0, 8))

        # Calculate the limits of each input parameter 
        bounds = [[min(i), max(i)] for i in self.input_data.values()]
        limits = np.array(bounds)
        self.limit_range = limits[:, 1] - limits[:, 0]

        # Vertically stack the x/y inputs and then transpose them 
        infos = np.vstack((xt[:, self.x_index], xt[:, self.y_index])).transpose()
        points = xt.copy()
        # Set the first two columns of the points array to x/y inputs, respectively 
        points[:, self.x_index] = self.input_point_list[self.x_index]
        points[:, self.y_index] = self.input_point_list[self.y_index]
        points = np.divide(points, self.limit_range)
        tree = cKDTree(points)
        dist_limit = np.linalg.norm(self.dist_range * self.limit_range)
        scaled_x0 = np.divide(self.input_point_list, self.limit_range)
        # Query the nearest neighbors tree for the closest points to the scaled x0 array
        dists, idx = tree.query(scaled_x0, k=len(xt), distance_upper_bound=dist_limit)
        idx = idx[idx != len(xt)]

        # info contains:
        # [x_value, y_value, ND-distance, func_value, alpha]

        data = np.zeros((len(idx), 5))
        for dist_index, i in enumerate(idx):
            if i != len(xt):
                info = np.ones((5))
                info[0:2] = infos[i, :]
                info[2] = dists[dist_index] / dist_limit
                info[3] = yt[i, output_variable]
                info[4] = (1. - info[2] / self.dist_range) ** 0.5
                data[dist_index] = info

        return data

    def stack_outputs(self, outputs_dict):

        """ Stack the values of a dictionary

        Parameters:
        outputs_dict (dict): Dictionary of outputs

        Returns:
        array: np.stack of values

        """
        output_lists_to_stack = []
        for values in outputs_dict.values():
            output_lists_to_stack.append(np.asarray(values))

        return np.stack(output_lists_to_stack, axis=-1)
