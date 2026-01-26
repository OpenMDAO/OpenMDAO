module MyParaboloidPackage

using ADTypes: ADTypes
using ComponentArrays: ComponentVector
using ForwardDiff: ForwardDiff
using OpenMDAOCore: OpenMDAOCore

function f_paraboloid!(Y, X, params)
    x = @view(X[:x])
    y = @view(X[:y])
    f_xy = @view(Y[:f_xy])
    @. f_xy = (x - 3.0)^2 + x*y + (y + 4.0)^2 - 3.0
    return nothing
end

function get_paraboloid_comp()
    ad_backend = ADTypes.AutoForwardDiff()
    X_ca = ComponentVector(x=1.0, y=1.0)
    Y_ca = ComponentVector(f_xy=0.0)
    comp = OpenMDAOCore.DenseADExplicitComp(ad_backend, f_paraboloid!, Y_ca, X_ca)
    return comp
end

end # module MyParaboloidPackage
