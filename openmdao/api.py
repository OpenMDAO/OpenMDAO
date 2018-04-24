"""Key OpenMDAO classes can be imported from here."""

# Core
from openmdao.core.problem import Problem
from openmdao.core.group import Group
from openmdao.core.parallel_group import ParallelGroup
from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.core.implicitcomponent import ImplicitComponent
from openmdao.core.indepvarcomp import IndepVarComp
from openmdao.core.analysis_error import AnalysisError

# Components
from openmdao.components.balance_comp import BalanceComp
from openmdao.components.cross_product_comp import CrossProductComp
from openmdao.components.dot_product_comp import DotProductComp
from openmdao.components.external_code import ExternalCode
from openmdao.components.exec_comp import ExecComp
from openmdao.components.linear_system_comp import LinearSystemComp
from openmdao.components.matrix_vector_product_comp import MatrixVectorProductComp
from openmdao.components.meta_model_unstructured import MetaModelUnStructured
from openmdao.components.multifi_meta_model import MultiFiMetaModelUnStructured
from openmdao.components.meta_model_structured import MetaModelStructured
from openmdao.components.meta_model_unstructured import MetaModel
from openmdao.components.multifi_meta_model import MultiFiMetaModel

# Solvers
from openmdao.solvers.linear.linear_block_gs import LinearBlockGS
from openmdao.solvers.linear.linear_block_jac import LinearBlockJac
from openmdao.solvers.linear.direct import DirectSolver
from openmdao.solvers.linear.petsc_ksp import PETScKrylov, PetscKSP
from openmdao.solvers.linear.linear_runonce import LinearRunOnce
from openmdao.solvers.linear.scipy_iter_solver import ScipyKrylov, ScipyIterativeSolver
from openmdao.solvers.linear.user_defined import LinearUserDefined
from openmdao.solvers.linesearch.backtracking import ArmijoGoldsteinLS
from openmdao.solvers.linesearch.backtracking import BoundsEnforceLS
from openmdao.solvers.nonlinear.nonlinear_block_gs import NonlinearBlockGS
from openmdao.solvers.nonlinear.nonlinear_block_jac import NonlinearBlockJac
from openmdao.solvers.nonlinear.newton import NewtonSolver
from openmdao.solvers.nonlinear.nonlinear_runonce import NonlinearRunOnce, NonLinearRunOnce

# Surrogate Models
from openmdao.surrogate_models.kriging import KrigingSurrogate, FloatKrigingSurrogate
from openmdao.surrogate_models.multifi_cokriging import MultiFiCoKrigingSurrogate, \
    FloatMultiFiCoKrigingSurrogate
from openmdao.surrogate_models.nearest_neighbor import NearestNeighbor
from openmdao.surrogate_models.response_surface import ResponseSurface
from openmdao.surrogate_models.surrogate_model import SurrogateModel, \
    MultiFiSurrogateModel

from openmdao.utils.find_cite import print_citations

# Vectors
from openmdao.vectors.default_vector import DefaultVector
try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None

# Developer Tools
from openmdao.devtools.problem_viewer.problem_viewer import view_model
from openmdao.devtools.viewconns import view_connections

# Derivative Specification
from openmdao.jacobians.assembled_jacobian import AssembledJacobian, \
    DenseJacobian, COOJacobian, CSRJacobian, CSCJacobian

# Drivers
try:
    from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver
except ImportError:
    pass
from openmdao.drivers.scipy_optimizer import ScipyOptimizer, ScipyOptimizeDriver
from openmdao.drivers.genetic_algorithm_driver import SimpleGADriver

# System-Building Tools
from openmdao.utils.options_dictionary import OptionsDictionary

# Recorders
from openmdao.recorders.sqlite_recorder import SqliteRecorder
from openmdao.recorders.case_reader import CaseReader

# set up tracing or memory profiling if env vars are set.
import os
if os.environ.get('OPENMDAO_TRACE'):
    from openmdao.devtools.itrace import setup, start
    ret = bool(os.environ.get('OPENMDAO_TRACE_RETURN'))
    setup(os.environ['OPENMDAO_TRACE'], show_return=ret)
    start()
elif os.environ.get('OPENMDAO_PROF_MEM'):
    from openmdao.devtools.iprof_mem import setup, start
    setup(os.environ['OPENMDAO_PROF_MEM'])
    start()
