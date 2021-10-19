import numpy as np
from ipytree import Tree, Node
from ipywidgets import Label, FloatText, HBox, VBox, Output, Button

from openmdao.core.component import Component
from openmdao.core.constants import _SetupStatus



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
        self._output = Output(layout={'border': '1px solid black', 'width': '60%'})

    def setup(self):
        # output = Output(layout={'border': '1px solid black', 'width': '60%'})
        print('setup with autoreload 2', self._output)
        self._output.append_stdout('gleep\n')

        if not (self._prob._metadata and self._prob._metadata['setup_status'] > _SetupStatus.PRE_SETUP):
            raise RuntimeError(f"The problem set values "
                                "GUI cannot be created before Problem.setup is called.")

        # self._tree = Tree(stripes=True, multiple_selection=True, layout={'border': '1px solid black', 'width' :'60%'})
        self._tree = Tree(stripes=True, multiple_selection=True, layout={'border': '1px solid black', 'width' :'30%'})
        self._value_widget_box = VBox([Label("Model Variables") ,], layout={'border': '1px solid black', 'width' :'50'})
        self.set_vars_from_model_with_initial_list(self._prob.model, self._tree,
                                                   vars_to_set=self._vars_to_set )
        # if self._vars_to_set:
        #     self.set_vars_from_user_input(self._vars_to_set, self._tree)
        # else:
        #     self.set_vars_from_model(self._prob.model, self._tree)
        self._tree.observe(self.on_selected_change, names='selected_nodes')
        # self.ui_widget = HBox([self._tree, self._value_widget_box, self._output])
        self.ui_widget = HBox([self._tree, self._value_widget_box])

    # def set_vars_from_model(self, sys, node):
    #     if sys.name == '_auto_ivc':
    #         return
    #     name = sys.name if sys.name else 'root'
    #     new_node = Node(name)
    #     node.add_node(new_node)
    #
    #     if isinstance(sys, Component):
    #         input_varnames = list(sys._var_allprocs_prom2abs_list['input'].keys())
    #         new_node.icon = 'plug'
    #         for input_varname in input_varnames:
    #             input_node = Node(input_varname)
    #             input_node.icon = 'signal'
    #             new_node.add_node(input_node)
    #     else:
    #         new_node.icon = 'envelope-open'
    #         for s in sys._subsystems_myproc:
    #             self.set_vars_from_model(s, new_node)

    def set_vars_from_model_with_initial_list(self, sys, node, vars_to_set=None):
        # Create the model tree selector widget
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
                input_node.icon = 'signal'
                new_node.add_node(input_node)

                if vars_to_set:
                    if input_varname in vars_to_set:

                        self.add_value_widget_with_component(sys, input_varname, input_node)
                else:
                    self.add_value_widget_with_component(sys, input_varname, input_node)

        else:
            new_node.icon = 'envelope-open'
            for s in sys._subsystems_myproc:
                self.set_vars_from_model_with_initial_list(s, new_node, vars_to_set)

        # Add initial value widgets if given by user
        # if vars_to_set:
        #     for var_name in vars_to_set:
        #         self.add_value_widget(var_name)


    # def set_vars_from_user_input(self, vars_to_set, node):
    #     for var_name in vars_to_set:
    #         self.add_value_widget(var_name)
    #
    def add_value_widget_with_component(self, comp, var_name, tree_node):
        self._output.append_stdout(f'add_value_widget_with_component: {var_name}\n')
        if var_name in self.get_widget_var_names():
            return # already there
        val = self._prob[var_name]

        inputs_metadata = comp.get_io_metadata(('input',), ['units', ],
                                     get_remote=True,
                                     return_rel_names=False)

        full_var_name = f"{comp.pathname}.{var_name}"

        metadata = inputs_metadata[full_var_name]
        units = metadata['units']
        if not units:
            units = "None"

        if isinstance(val, np.ndarray):
            if val.size > 1:
                return # skip arrays for now
            val = val.item()

        # Value widget
        val_widget = FloatText(
            value=self._prob[var_name],
            description=var_name,
            disabled=False
        )
        val_widget.observe(self.update_prob_val, 'value')


        # Units label
        units_label = Label(units, layout={'border': '1px solid black',
                                                        'width' :'60px',
                                                        'display':'flex',
                                                        'justify_content':'center'})

        # remove button widget
        remove_button = Button(description="X", layout={'border': '1px solid black',
                                                        'width' :'20px',
                                                        'display':'flex',
                                                        'justify_content':'center'})
        remove_button._var_name = var_name # so each Button instance know what variable it is
        # associated with
        remove_button.on_click(self.remove_val_widget)

        # Put them all together
        # val_and_remove_widget = HBox([val_widget, remove_button])
        val_and_remove_widget = HBox([val_widget, units_label, remove_button])

        val_and_remove_widget._tree_node = tree_node

        tree_node.icon_style = 'warning'


        self._value_widget_box.children += (val_and_remove_widget,)

    # def add_value_widget(self, var_name):
    #     if var_name in self.get_widget_var_names():
    #         return # already there
    #     val = self._prob[var_name]
    #
    #     if isinstance(val, np.ndarray):
    #         if val.size > 1:
    #             return # skip arrays for now
    #         val = val.item()
    #     val_widget = FloatText(
    #         value=self._prob[var_name],
    #         description=var_name,
    #         disabled=False
    #     )
    #     val_widget.observe(self.update_prob_val, 'value')
    #     remove_button = Button(description="X", layout={'border': '1px solid black',
    #                                                     'width' :'20px',
    #                                                     'display':'flex',
    #                                                     'justify_content':'center'})
    #     remove_button._var_name = var_name # so each Button instance know what variable it is
    #     # associated with
    #     remove_button.on_click(self.remove_val_widget)
    #     val_widget.observe(self.update_prob_val, 'value')
    #     val_and_remove_widget = HBox([val_widget, remove_button])
    #     self._value_widget_box.children += (val_and_remove_widget,)


    def on_selected_change(self, change):
        self._output.append_stdout(f"on_selected_change\n")
        self._output.append_stdout(f"change['new'][0]: {change['new'][0]}\n")
        self._output.append_stdout(f"change['new'][0]._comp: {change['new'][0]._comp}\n")
        comp = change['new'][0]._comp
        change['new'][0].icon_style = 'warning'
        var_name = change['new'][0].name
        self._output.append_stdout(f"{self.get_widget_var_names()}\n")
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
            associated_tree_node.icon_style = 'default'

    def update_prob_val(self, change):
        self._prob[change['owner'].description] = change['new']

    def get_widget_var_names(self):
        var_names = []
        for box in self._value_widget_box.children[1:]:
            float_text_widget = box.children[0]
            var_names.append(float_text_widget.description)
        return var_names

    def display(self):
        return self.ui_widget