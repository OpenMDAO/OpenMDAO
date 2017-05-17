.. index:: multiple outputs metamodel

Modeling Multiple Outputs
==================================

This tutorial is a short demonstration of how to construct a MetaModel of a
component with multiple outputs. This tutorial builds off of the
:ref:`single-output tutorial <Using-a-MetaModel-Component>`, with
modifications for multiple outputs in a component.

We created a new component called ``Trig()``. This component has one input and two
outputs, both of which will be mimicked by the MetaModel.

.. testcode:: Mult_out_parts

    from openmdao.main.api import Assembly, Component, SequentialWorkflow, set_as_top
    from math import sin, cos

    from openmdao.lib.datatypes.api import Float
    from openmdao.lib.drivers.api import DOEdriver
    from openmdao.lib.doegenerators.api import FullFactorial, Uniform
    from openmdao.lib.components.api import MetaModel
    from openmdao.lib.casehandlers.api import DBCaseRecorder
    from openmdao.lib.surrogatemodels.api import LogisticRegression, FloatKrigingSurrogate


    class Trig(Component):

        x = Float(0,iotype="in",units="rad")

        f_x_sin = Float(0.0,iotype="out")
        f_x_cos = Float(0.0,iotype="out")

        def execute(self):
            self.f_x_sin = .5*sin(self.x)
            self.f_x_cos = .5*cos(self.x)

This next section differs from the the previous example in that there are two surrogate models,  one
specified for each of the outputs. Note that each of the outputs had been assigned  a specific
surrogate model, a logistic regression for sin, and a Kriging Surrogate for cos. In this case,  no
default was set at all.

The parameter `x` still needs to be added only once in this case, since the same input
is being evaluated for both outputs.


.. testcode:: Mult_out_parts

    class Simulation(Assembly):

        def configure(self):

            # Our component to be meta-modeled
            self.add("trig_calc", Trig())

           # Create meta_model for two responsese
            self.add("trig_meta_model", MetaModel(params = ('x', ),
                                                  responses = ('f_x_sin', 'f_x_cos')))

            # Use Kriging for the f_x output
            self.trig_meta_model.surrogates['f_x_sin'] = LogisticRegression()
            self.trig_meta_model.surrogates['f_x_cos'] = FloatKrigingSurrogate()

            # Training the MetaModel
            self.add("DOE_Trainer", DOEdriver())
            self.DOE_Trainer.DOEgenerator = FullFactorial()
            self.DOE_Trainer.DOEgenerator.num_levels = 20
            self.DOE_Trainer.add_parameter("trig_calc.x", low=0, high=20)
            self.DOE_Trainer.add_response('trig_calc.f_x_sin')
            self.DOE_Trainer.add_response('trig_calc.f_x_cos')

            # Pass training data to the meta model.
            self.connect('DOE_Trainer.case_inputs.trig_calc.x', 'trig_meta_model.params.x')
            self.connect('DOE_Trainer.case_outputs.trig_calc.f_x_sin', 'trig_meta_model.responses.f_x_sin')
            self.connect('DOE_Trainer.case_outputs.trig_calc.f_x_cos', 'trig_meta_model.responses.f_x_cos')

            #MetaModel Validation
            self.add("DOE_Validate", DOEdriver())
            self.DOE_Validate.DOEgenerator = Uniform()
            self.DOE_Validate.DOEgenerator.num_samples = 20
            self.DOE_Validate.add_parameter(("trig_meta_model.x", "trig_calc.x"),
                                            low=0, high=20)
            self.DOE_Validate.add_response("trig_calc.f_x_sin")
            self.DOE_Validate.add_response("trig_calc.f_x_cos")
            self.DOE_Validate.add_response("trig_meta_model.f_x_sin")
            self.DOE_Validate.add_response("trig_meta_model.f_x_cos")

            #Iteration Hierarchy
            self.driver.workflow.add(['DOE_Trainer', 'DOE_Validate'])
            self.DOE_Trainer.workflow.add('trig_calc')
            self.DOE_Validate.workflow.add(['trig_calc', 'trig_meta_model'])


The iteration hierarchy is structurally the same as it would be with one
output. Even though there are multiple surrogate models for multiple outputs,
they are still contained within only one MetaModel component. So once again
there is the ``trig_calc`` MetaModel component separately added to each
workflow and the MetaModel component being added to the validation stage so
that comparative values may be generated.

In printing the information we have now included all four of the outputs. For
a Kriging Surrogate model, the answer is normally returned as a normal
distribution (Kriging Surrogate predicts both a mean and a standard deviation
for a given input). However, here we have slotted the
``FloatKrigingSurrogate``, which just returns the mean (or mu).

.. testcode:: Mult_out_parts

    if __name__ == "__main__":

        sim = set_as_top(Simulation())
        sim.run()

        #This is how you can access any of the data
        train_inputs = sim.DOE_Trainer.case_inputs.trig_calc.x
        train_actual_sin = sim.DOE_Trainer.case_outputs.trig_calc.f_x_sin
        train_actual_cos = sim.DOE_Trainer.case_outputs.trig_calc.f_x_cos
        inputs = sim.DOE_Validate.case_inputs.trig_meta_model.x
        actual_sin = sim.DOE_Validate.case_outputs.trig_calc.f_x_sin
        actual_cos = sim.DOE_Validate.case_outputs.trig_calc.f_x_cos
        predicted_sin = sim.DOE_Validate.case_outputs.trig_meta_model.f_x_sin
        predicted_cos = sim.DOE_Validate.case_outputs.trig_meta_model.f_x_cos

        for a,b,c,d in zip(actual_sin, predicted_sin, actual_cos, predicted_cos):
            print "%1.3f, %1.3f, %1.3f, %1.3f"%(a, b, c, d)

To view this example, and try running and modifying the code for yourself, you can download it here:
:download:`multi_outs.py </../examples/openmdao.examples.metamodel_tutorial/openmdao/examples/metamodel_tutorial/multi_outs.py>`.
