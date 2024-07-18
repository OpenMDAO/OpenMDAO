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
from openmdao.components.add_subtract_comp import AddSubtractComp
from openmdao.components.balance_comp import BalanceComp
from openmdao.components.cross_product_comp import CrossProductComp
from openmdao.components.dot_product_comp import DotProductComp
from openmdao.components.eq_constraint_comp import EQConstraintComp
from openmdao.components.exec_comp import ExecComp
from openmdao.components.explicit_func_comp import ExplicitFuncComp
from openmdao.components.implicit_func_comp import ImplicitFuncComp
from openmdao.components.input_resids_comp import InputResidsComp
from openmdao.components.external_code_comp import ExternalCodeComp
from openmdao.components.external_code_comp import ExternalCodeImplicitComp
from openmdao.components.ks_comp import KSComp
from openmdao.components.linear_system_comp import LinearSystemComp
from openmdao.components.matrix_vector_product_comp import MatrixVectorProductComp
from openmdao.components.meta_model_structured_comp import MetaModelStructuredComp
from openmdao.components.meta_model_semi_structured_comp import MetaModelSemiStructuredComp
from openmdao.components.meta_model_unstructured_comp import MetaModelUnStructuredComp
from openmdao.components.spline_comp import SplineComp
from openmdao.components.multifi_meta_model_unstructured_comp \
    import MultiFiMetaModelUnStructuredComp
from openmdao.components.mux_comp import MuxComp
from openmdao.components.vector_magnitude_comp import VectorMagnitudeComp
from openmdao.components.submodel_comp import SubmodelComp


# Solvers
from openmdao.solvers.linear.linear_block_gs import LinearBlockGS
from openmdao.solvers.linear.linear_block_jac import LinearBlockJac
from openmdao.solvers.linear.direct import DirectSolver
from openmdao.solvers.linear.petsc_ksp import PETScKrylov
from openmdao.solvers.linear.linear_runonce import LinearRunOnce
from openmdao.solvers.linear.scipy_iter_solver import ScipyKrylov
from openmdao.solvers.linear.user_defined import LinearUserDefined
from openmdao.solvers.linesearch.backtracking import ArmijoGoldsteinLS
from openmdao.solvers.linesearch.backtracking import BoundsEnforceLS
from openmdao.solvers.nonlinear.broyden import BroydenSolver
from openmdao.solvers.nonlinear.nonlinear_block_gs import NonlinearBlockGS
from openmdao.solvers.nonlinear.nonlinear_block_jac import NonlinearBlockJac
from openmdao.solvers.nonlinear.newton import NewtonSolver
from openmdao.solvers.nonlinear.nonlinear_runonce import NonlinearRunOnce

# Surrogate Models
from openmdao.surrogate_models.kriging import KrigingSurrogate
from openmdao.surrogate_models.multifi_cokriging import MultiFiCoKrigingSurrogate
from openmdao.surrogate_models.nearest_neighbor import NearestNeighbor
from openmdao.surrogate_models.response_surface import ResponseSurface
from openmdao.surrogate_models.surrogate_model import SurrogateModel, \
    MultiFiSurrogateModel

from openmdao.utils.coloring import display_coloring
from openmdao.utils.indexer import slicer, indexer
from openmdao.utils.find_cite import print_citations
from openmdao.utils.spline_distributions import cell_centered
from openmdao.utils.spline_distributions import sine_distribution
from openmdao.utils.spline_distributions import node_centered

# Vectors
from openmdao.vectors.default_vector import DefaultVector
try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:  # pragma: no cover
    PETScVector = None

# Drivers
from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver
from openmdao.drivers.scipy_optimizer import ScipyOptimizeDriver
from openmdao.drivers.genetic_algorithm_driver import SimpleGADriver
from openmdao.drivers.differential_evolution_driver import DifferentialEvolutionDriver
from openmdao.drivers.doe_driver import DOEDriver
from openmdao.drivers.doe_generators import ListGenerator, CSVGenerator, UniformGenerator, \
    FullFactorialGenerator, PlackettBurmanGenerator, BoxBehnkenGenerator, LatinHypercubeGenerator, \
    GeneralizedSubsetGenerator

# System-Building Tools
from openmdao.utils.options_dictionary import OptionsDictionary

# Recorders
from openmdao.recorders.sqlite_recorder import SqliteRecorder
from openmdao.recorders.case_reader import CaseReader

# Visualizations
from openmdao.visualization.n2_viewer.n2_viewer import n2
from openmdao.visualization.connection_viewer.viewconns import view_connections
from openmdao.visualization.partial_deriv_plot import partial_deriv_plot
from openmdao.visualization.timing_viewer.timer import timing_context
from openmdao.visualization.timing_viewer.timing_viewer import view_timing, view_timing_dump, \
    view_MPI_timing
from openmdao.visualization.options_widget import OptionsWidget
from openmdao.visualization.case_viewer.case_viewer import CaseViewer
from openmdao.visualization.tables.table_builder import generate_table

# Notebook Utils
from openmdao.utils.notebook_utils import notebook_mode, display_source, show_options_table, cite

# Units
from openmdao.utils.units import convert_units, unit_conversion

# Warning Options
from openmdao.utils.om_warnings import issue_warning, reset_warnings, OpenMDAOWarning, \
    SetupWarning, DistributedComponentWarning, CaseRecorderWarning, \
    DriverWarning, CacheWarning, PromotionWarning, UnusedOptionWarning, DerivativesWarning, \
    MPIWarning, UnitsWarning, SolverWarning, OMDeprecationWarning, \
    OMInvalidCheckDerivativesOptionsWarning

# Utils
from openmdao.utils.general_utils import wing_dbg, env_truthy
from openmdao.utils.array_utils import shape_to_len
from openmdao.utils.jax_utils import register_jax_component

# Reports System
from openmdao.utils.reports_system import register_report, unregister_report, get_reports_dir, \
    list_reports, clear_reports, set_reports_dir

import os

wing_dbg()

# set up tracing or memory profiling if env vars are set.
if env_truthy('OPENMDAO_TRACE'):  # pragma: no cover
    from openmdao.devtools.itrace import setup, start
    setup(os.environ['OPENMDAO_TRACE'])
    start()
elif env_truthy('OPENMDAO_PROF_MEM'):  # pragma: no cover
    from openmdao.devtools.iprof_mem import setup, start
    setup(os.environ['OPENMDAO_PROF_MEM'])
    start()


if env_truthy('FLUSH_PRINT'):  # pragma: no cover
    import builtins
    _oldprint = builtins.print

    def _flushprint(*args, **kwargs):
        kwargs['flush'] = True
        _oldprint(*args, **kwargs)

    builtins.print = _flushprint
