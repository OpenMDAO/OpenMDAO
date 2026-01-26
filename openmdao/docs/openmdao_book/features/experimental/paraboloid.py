import os

# Create a new Julia module that will hold all the Julia code imported into this Python module.
import juliacall
jl = juliacall.newmodule("ParaboloidComponentsStub")

# Get the directory this file is in, then include the `paraboloid.jl` Julia source code.
d = os.path.dirname(os.path.abspath(__file__))
jl.include(os.path.join(d, "paraboloid.jl"))
# Now we have access to everything in `paraboloid.jl` in the `jl` object.

get_paraboloid_comp = jl.get_paraboloid_comp
