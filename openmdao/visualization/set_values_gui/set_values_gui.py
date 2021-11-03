import numpy as np
from ipytree import Tree, Node
from ipywidgets import Label, HBox, VBox, Output, Button, Text

from openmdao.core.component import Component
from openmdao.core.constants import _SetupStatus

from openmdao.utils.units import _find_unit, _UNIT_LIB
def find_compatible_units(input_units_string):
    input_units = _find_unit(input_units_string)
    compatible_units = list()
    for units_string in _UNIT_LIB.unit_table.keys():
        units = _find_unit(units_string)
        if input_units.is_compatible(units):
            compatible_units.append(units_string)
    return compatible_units

class SetValuesUI(object):
    def __init__(self, prob, vars_to_set=None):
        """
        Initialize attributes.
        """
        self._prob = prob
        self._vars_to_set = vars_to_set
        self._value_widget_box = None
        self._tree = None
        self.ui_widget = None
        self._refresh_in_progress = False
        self._output = Output(layout={'border': '1px solid black', 'width': '35%'})

    def setup(self):
        if not (self._prob._metadata and self._prob._metadata['setup_status'] > _SetupStatus.PRE_SETUP):
            raise RuntimeError(f"The problem set values "
                                "GUI cannot be created before Problem.setup is called.")

        self._tree = Tree(stripes=True, multiple_selection=True, layout={'border': '1px solid black', 'width' :'30%'})
        self._value_widget_box = VBox([Label("Model Variables") ,], layout={'border': '1px solid black', 'width' :'50'})
        self.set_vars_from_model_with_initial_list_v2(self._prob.model, self._tree, 0,
                                                   vars_to_set=self._vars_to_set )
        self._tree.observe(self.on_selected_change, names='selected_nodes')

        refresh_button = Button(description="Refresh", layout={'border': '1px solid black',
                                                        'width' :'80px',
                                                        'display':'flex',
                                                        'justify_content':'center'})
        refresh_button.on_click(self.refresh_ui_with_problem_data)
        self.ui_widget = HBox([self._tree, self._value_widget_box, refresh_button,self._output])

    def set_vars_from_model_with_initial_list_v2(self, sys, node, depth, vars_to_set=None):
        # Create the model tree selector widget
        # Can use icons from here https://fontawesome.com/v5.15/icons?d=gallery&p=2&m=free
        if sys.name == '_auto_ivc':
            return

        name = sys.name if sys.name else 'root'
        new_node = Node(name)
        node.add_node(new_node)

        if depth > 1:
            new_node.opened = False

        model = sys._problem_meta['model_ref']()

        var_names_at_this_level = get_var_names_at_this_level(model, sys)

        if isinstance(sys, Component):
            input_varnames = var_names_at_this_level
            new_node.icon = 'plug'


            for input_varname in input_varnames:
                input_node = Node(input_varname)
                input_node._comp = sys
                input_node.icon = 'eye-slash'  # far fa-eye    OR eye  OR eye-slash or square OR square-full OR toggle-off toggle-on
                new_node.add_node(input_node)

                if vars_to_set is not None:
                    # print("Component vars_to_set is not None")
                    if input_varname in vars_to_set:
                        # print(f"Component input_varname in vars_to_set for {input_varname}")
                        input_node.icon_style = 'success'
                        self.add_value_widget_with_component(sys, input_varname, input_node)
                else:
                    # print("Component vars_to_set is None")
                    input_node.icon_style = 'success'
                    self.add_value_widget_with_component(sys, input_varname, input_node)

        else:

            input_varnames = var_names_at_this_level
            new_node.icon = 'plug'
            for input_varname in input_varnames:
                input_node = Node(input_varname)
                input_node._comp = sys
                input_node.icon = 'eye-slash'  # far fa-eye    OR eye  OR eye-slash or square OR square-full OR toggle-off toggle-on
                new_node.add_node(input_node)

                if vars_to_set is not None:
                    # print("Group vars_to_set is not None")
                    if input_varname in vars_to_set:
                        # print(f"Group input_varname in vars_to_set for {input_varname}")
                        input_node.icon_style = 'success'
                        self.add_value_widget_with_component(sys, input_varname, input_node)
                else:
                    # print("Group vars_to_set is None")
                    input_node.icon_style = 'success'
                    self.add_value_widget_with_component(sys, input_varname, input_node)

            new_node.icon = 'envelope-open'
            for s in sys._subsystems_myproc:
                import time
                time.sleep(0.1) # To slow down the messages which cause issue with ipwidgets
                self.set_vars_from_model_with_initial_list_v2(s, new_node, depth + 1, vars_to_set )


    def set_vars_from_model_with_initial_list_v1(self, sys, node, vars_to_set=None):
        # Create the model tree selector widget
        # Can use icons from here https://fontawesome.com/v5.15/icons?d=gallery&p=2&m=free
        if sys.name == '_auto_ivc':
            return
        name = sys.name if sys.name else 'root'
        new_node = Node(name)
        node.add_node(new_node)

        if isinstance(sys, Component):
            input_varnames = list(sys._var_allprocs_prom2abs_list['input'].keys())
            new_node.icon = 'plug'
            for input_varname in input_varnames:
                input_node = Node(input_varname)
                input_node._comp = sys
                input_node.icon = 'eye-slash'
                new_node.add_node(input_node)

                if vars_to_set and input_varname in vars_to_set:
                        input_node.icon_style = 'success'
                        self.add_value_widget_with_component(sys, input_varname, input_node)
                else:
                    input_node.icon_style = 'success'
                    self.add_value_widget_with_component(sys, input_varname, input_node)

        else:
            new_node.icon = 'envelope-open'
            for s in sys._subsystems_myproc:
                self.set_vars_from_model_with_initial_list_v1(s, new_node, vars_to_set)

    def add_value_widget_with_component(self, comp, var_name, tree_node):
        if var_name in self.get_widget_var_names():
            return # already there
        val = comp.get_val(var_name)

        inputs_metadata = comp.get_io_metadata(('input',), ['units', ],
                                     get_remote=True,
                                     return_rel_names=False)

        prom2abs_list = comp._var_allprocs_prom2abs_list['input']

        full_var_name = prom2abs_list[var_name][0]

        metadata = inputs_metadata[full_var_name]
        units = metadata['units']
        if not units:
            units = "None"

        if isinstance(val, np.ndarray):
            if val.size > 1:
                return # skip arrays for now
            val = val.item()

        # Value widget
        # val_widget = FloatText(
        val_widget = Text(
            value=str(val),
            description=var_name,
            disabled=False,
            continuous_update=False,
            step=None,
            layout = {'border': '1px solid black',
                      'width': '150px',
                      'display': 'flex',
                      'justify_content': 'center'}
        )
        val_widget.observe(self.update_prob_val, 'value')

        units_widget = Text(units,
                            continuous_update=False,
                            description_tooltip=var_name,
                            layout={'border': '1px solid black',
                                                        'width' :'60px',
                                                        'display':'flex',
                                                        'justify_content':'center'})
        units_widget._var_name = var_name
        units_widget.observe(self.update_prob_unit, 'value')

        # remove button widget
        remove_button = Button(description="X", layout={'border': '1px solid black',
                                                        'width' :'20px',
                                                        'display':'flex',
                                                        'justify_content':'center'})
        remove_button._var_name = var_name # so each Button instance know what variable it is
        # associated with
        remove_button.on_click(self.remove_val_widget)

        val_and_remove_widget = HBox([val_widget, units_widget, remove_button])

        val_and_remove_widget._tree_node = tree_node

        tree_node.icon = 'eye'

        self._value_widget_box.children += (val_and_remove_widget,)

    def on_selected_change(self, change):
        comp = change['new'][0]._comp
        change['new'][0].icon_style = 'info' # also try danger, success, info
        change['new'][0].icon = 'eye'
        var_name = change['new'][0].name
        if var_name in self.get_widget_var_names():
            change['new'][0].selected = False
            return

        change['new'][0].selected = False

        tree_node = change['new'][0]

        self.add_value_widget_with_component(comp, var_name, tree_node)

    def remove_val_widget(self, button):
        # also need to go through self._value_widget_box.children. Each is an HBox with a
        # FloatText widget
        box_to_remove = None
        for box in self._value_widget_box.children[1:]: # skip the first since it is the Label. The rest are HBoxes
            float_text_widget = box.children[0]
            if button._var_name == float_text_widget.description:
                box_to_remove = box
                break
        # Cannot use remove on tuples, which children are so
        if box_to_remove:
            self._value_widget_box.children = tuple \
                (box for box in self._value_widget_box.children if box != box_to_remove)
            box_to_remove.close()
            associated_tree_node = box_to_remove._tree_node
            # associated_tree_node.icon_style = 'default'
            associated_tree_node.icon = 'eye-slash'

    def refresh_ui_with_problem_data(self, button):
        print("refresh_ui_with_problem_data")
        self._refresh_in_progress = True
        # loop through self._value_widget_box.children
        for box in self._value_widget_box.children[1:]:
            val_widget = box.children[0]
            val = self._prob[val_widget.description]
            if isinstance(val, np.ndarray):
                if val.size > 1:
                    return  # skip arrays for now
                val = val.item()

            val_widget.value = str(val)
        self._refresh_in_progress = False

    def update_prob_val(self, change):
        if not self._refresh_in_progress: # no need to update problem since values from problem are being used to change the value in the widget
            # self._prob[change['owner'].description] = change['new']
            val = float(change['new']) if change['new'] else 0.0
            self._prob.set_val(change['owner'].description, val)

    def update_prob_unit(self, change):
        var_name = change['owner']._var_name
        units = change['new']
        val = self._prob.get_val(var_name)
        self._prob.set_val(var_name, val, units=units)

        # self._prob.model.set_input_defaults(change['owner']._var_name, units=change['new'])

    def get_widget_var_names(self):
        var_names = []
        for box in self._value_widget_box.children[1:]:
            float_text_widget = box.children[0]
            var_names.append(float_text_widget.description)
        return var_names

    def display(self):
        return self.ui_widget

def set_values_gui(prob, vars_to_set=None):
    ui = SetValuesUI(prob, vars_to_set=vars_to_set)
    ui.setup()
    return ui.ui_widget



def get_var_names_at_this_level(model, sys):

    if 'DESIGN.inlet.real_flow' == sys.pathname:
        print(sys)

    var_names_at_this_level = set()
    model_abs2prom = model._var_allprocs_abs2prom['input']
    if isinstance(sys, Component):
        for abs_name, prom_name in sys._var_allprocs_abs2prom['input'].items():
            promoted_name_from_top_level = model_abs2prom[abs_name]
            current_group_absolute_path = sys.pathname
            current_group_var_promoted_name = prom_name
            # if they're different you would know that someone above the current group promoted it
            current_group_pathname = f"{current_group_absolute_path}.{current_group_var_promoted_name}"
            if promoted_name_from_top_level == current_group_pathname:
                var_names_at_this_level.add(prom_name)
    else: # Group
        for abs_name, prom_name in sys._var_allprocs_abs2prom['input'].items():
            promoted_name_from_top_level = model_abs2prom[abs_name]
            current_group_absolute_path = sys.pathname
            current_group_var_promoted_name = prom_name
            # if they're different you would know that someone above the current group promoted it
            current_group_pathname = f"{current_group_absolute_path}.{current_group_var_promoted_name}"
            if "." in current_group_var_promoted_name:
                continue
            if promoted_name_from_top_level == current_group_pathname:
                var_names_at_this_level.add(prom_name)

    return var_names_at_this_level
