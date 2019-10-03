.. _feature_MultiFiMetaModelUnStructuredComp:

********************************
MultiFiMetaModelUnStructuredComp
********************************

`MultiFiMetaModelUnStructuredComp` is a component that allows you to create a surrogate model where
the training data has been gathered from sources with multiple levels of fidelity. This approach
can be beneficial when a high-fidelity model is expensive to evaluate but a low-fidelity model
exists that can be evaluated more efficiently at the cost of some accuracy; the main benefit comes
from replacing some of the expensive evaluation points with a larger set of cheaper points while
maintaining confidence in the overall prediction.

`MultiFiMetaModelUnStructuredComp` inherits from :ref:`MetaModelUnStructuredComp<feature_MetaModelUnStructuredComp>`, so its interface and
usage are mostly the same. However, it does not use the same SurrogateModels. The only available
SurrogateModel is the `MultiFiCoKrigingSurrogate`, which implements the Multi-Fidelity Co-Kriging
method as found in Scikit-Learn.

MultiFiMetaModelUnStructuredComp Options
----------------------------------------

.. embed-options::
    openmdao.components.multifi_meta_model_unstructured_comp
    MultiFiMetaModelUnStructuredComp
    options

Simple Example
--------------

The following example shows a `MultiFiMetaModelUnStructuredComp` used to model the 2D Branin
function, where the output is a function of two inputs, and we have pre-computed the training
point location and values at a variety of points using models with two different fidelity
levels. Adding an input or output named 'x' spanws entries in the "options" dictionary where the
training data can be specified. The naming convention is 'train:y' for the highest fidelity, and
'train:y_fi2' for the lowest fidelity level (or in the case of more than 2 fidelity levels, the
next highest level.)

.. embed-code::
    openmdao.components.tests.test_multifi_meta_model_unstructured_comp.MultiFiMetaModelFeatureTestCase.test_2_input_2_fidelity
    :layout: code, output