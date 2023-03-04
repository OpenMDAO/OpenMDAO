from setuptools import setup

from pathlib import Path
from io import open

import re

__version__ = re.findall(
    r"""__version__ = ["']+([0-9\.\-dev]*)["']+""",
    open('openmdao/__init__.py').read(),
)[0]

with open(Path(__file__).parent / "README.md", encoding="utf-8") as f:
    long_description = f.read()


optional_dependencies = {
    'docs': [
        'matplotlib',
        'numpydoc>=1.1',
        'jupyter-book==0.14',
        'sphinx-sitemap',
        'ipyparallel'
    ],
    'doe': [
        'pyDOE2'
    ],
    'notebooks': [
        'notebook',
        'ipympl'
    ],
    'visualization': [
        'bokeh>=1.3.4',
        'matplotlib',
        'colorama'
    ],
    'test': [
        'parameterized',
        'numpydoc>=1.1',
        'pycodestyle>=2.4.0',
        'pydocstyle>=2.0.0',
        'testflo>=1.3.6',
        'websockets>8',
        'aiounittest',
        'playwright>=1.20',
        'num2words'
    ]
}

# Add an optional dependency that concatenates all others
optional_dependencies['all'] = sorted([
    dependency
    for dependencies in optional_dependencies.values()
    for dependency in dependencies
])

setup(
    name='openmdao',
    version=__version__,
    description="OpenMDAO framework infrastructure",
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: Implementation :: CPython',
    ],
    keywords='optimization multidisciplinary multi-disciplinary analysis',
    author='OpenMDAO Team',
    author_email='openmdao@openmdao.org',
    url='http://openmdao.org',
    license='Apache License, Version 2.0',
    packages=[
        'openmdao',
        'openmdao.approximation_schemes',
        'openmdao.code_review',
        'openmdao.components',
        'openmdao.components.interp_util',
        'openmdao.core',
        'openmdao.devtools',
        'openmdao.devtools.iprofile_app',
        'openmdao.docs',
        'openmdao.docs.openmdao_book',
        'openmdao.drivers',
        'openmdao.error_checking',
        'openmdao.jacobians',
        'openmdao.matrices',
        'openmdao.proc_allocators',
        'openmdao.recorders',
        'openmdao.solvers',
        'openmdao.solvers.linear',
        'openmdao.solvers.linesearch',
        'openmdao.solvers.nonlinear',
        'openmdao.surrogate_models',
        'openmdao.surrogate_models.nn_interpolators',
        'openmdao.test_suite',
        'openmdao.test_suite.components',
        'openmdao.test_suite.groups',
        'openmdao.test_suite.test_examples',
        'openmdao.test_suite.test_examples.beam_optimization',
        'openmdao.test_suite.test_examples.beam_optimization.components',
        'openmdao.test_suite.test_examples.cannonball',
        'openmdao.test_suite.test_examples.meta_model_examples',
        'openmdao.utils',
        'openmdao.vectors',
        'openmdao.visualization',
        'openmdao.visualization.case_viewer',
        'openmdao.visualization.connection_viewer',
        'openmdao.visualization.inputs_report',
        'openmdao.visualization.timing_viewer',
        'openmdao.visualization.scaling_viewer',
        'openmdao.visualization.n2_viewer',
        'openmdao.visualization.meta_model_viewer',
        'openmdao.visualization.opt_report',
        'openmdao.visualization.tables',
    ],
    package_data={
        'openmdao.devtools': ['*.wpr', ],
        'openmdao.visualization': [
            'common/libs/*.js',
            'common/style/*.css'
        ],
        'openmdao.visualization.n2_viewer': [
            'assets/*png*',
            'libs/*.js',
            'src/*.js',
            'gen/*.js',
            'style/*',
            'tests/*.js',
            'tests/*.json',
            'tests/*.html',
            'tests/gui_test_models/*.py',
            '*.html'
        ],
        'openmdao.visualization.connection_viewer': [
            '*.html',
        ],
        'openmdao.visualization.tables': [
            '*.template',
        ],
        'openmdao.visualization.timing_viewer': [
            '*.html',
        ],
        'openmdao.visualization.scaling_viewer': [
            '*.html',
        ],
        'openmdao.visualization.opt_report': [
            '*.html',
        ],
        'openmdao.visualization.meta_model_viewer': [
            'tests/known_data_point_files/*.csv',
        ],
        'openmdao.devtools.iprofile_app': [
            'static/*.html',
            'templates/*.html'
        ],
        'openmdao.docs': ['*.py', 'openmdao_book/test/*.py', 'openmdao_book/other/*.py',
                          'openmdao_book/*.md',
                          'openmdao_book/*/*.md',
                          'openmdao_book/*.ipynb',
                          'openmdao_book/*/*.ipynb',
                          'openmdao_book/*/*/*.ipynb',
                          'openmdao_book/*/*/*/*.ipynb'],
        'openmdao.recorders': ['tests/legacy_sql/*.sql'],
        'openmdao.utils': ['unit_library.ini', 'scaffolding_templates/*'],
        'openmdao.test_suite': [
            '*.py',
            '*/*.py',
            'matrices/*.npz'
        ],
        'openmdao': ['*/tests/*.py', '*/*/tests/*.py', '*/*/*/tests/*.py']
    },
    python_requires=">=3.7",
    install_requires=[
        'networkx>=2.0',
        'numpy',
        'scipy',
        'requests',
        'packaging'
    ],
    entry_points={
        'console_scripts': [
            'wingproj=openmdao.devtools.wingproj:run_wing',
            'webview=openmdao.utils.webview:webview_argv',
            'run_om_test=openmdao.devtools.run_test:run_test',
            'reset_notebook=openmdao.devtools.docutils:reset_notebook_cmd',
            'openmdao=openmdao.utils.om:openmdao_cmd',
        ],
        'openmdao_case_reader': [
            'sqlitereader=openmdao.recorders.sqlite_reader:SqliteCaseReader',
        ],
        'openmdao_case_recorder': [
            'sqliterecorder=openmdao.recorders.sqlite_recorder:SqliteRecorder',
        ],
        'openmdao_component': [
            'addsubtractcomp=openmdao.components.add_subtract_comp:AddSubtractComp',
            'balancecomp=openmdao.components.balance_comp:BalanceComp',
            'crossproductcomp=openmdao.components.cross_product_comp:CrossProductComp',
            'demuxcomp=openmdao.components.demux_comp:DemuxComp',
            'dotproductcomp=openmdao.components.dot_product_comp:DotProductComp',
            'eqconstraintcomp=openmdao.components.eq_constraint_comp:EQConstraintComp',
            'execcomp=openmdao.components.exec_comp:ExecComp',
            'explicitfunccomp=openmdao.components.explicit_func_comp:ExplicitFuncComp',
            'implicitfunccomp=openmdao.components.implicit_func_comp:ImplicitFuncComp',
            'externalcodecomp=openmdao.components.external_code_comp:ExternalCodeComp',
            'externalcodeimplicitcomp=openmdao.components.external_code_comp:ExternalCodeImplicitComp',
            'kscomp=openmdao.components.ks_comp:KSComp',
            'linearsystemcomp=openmdao.components.linear_system_comp:LinearSystemComp',
            'matrixvectorproductcomp=openmdao.components.matrix_vector_product_comp:MatrixVectorProductComp',
            'metamodelstructuredcomp=openmdao.components.meta_model_structured_comp:MetaModelStructuredComp',
            'metamodelsemistructuredcomp=openmdao.components.meta_model_semi_structured_comp:MetaModelSemiStructuredComp',
            'metamodelunstructuredcomp=openmdao.components.meta_model_unstructured_comp:MetaModelUnStructuredComp',
            'multifimetamodelunstructuredcomp=openmdao.components.multifi_meta_model_unstructured_comp:MultiFiMetaModelUnStructuredComp',
            'muxcomp=openmdao.components.mux_comp:MuxComp',
            'splinecomp=openmdao.components.spline_comp:SplineComp',
            'vectormagnitudecomp=openmdao.components.vector_magnitude_comp:VectorMagnitudeComp',
            'indepvarcomp=openmdao.core.indepvarcomp:IndepVarComp',
        ],
        'openmdao_driver': [
            'doedriver=openmdao.drivers.doe_driver:DOEDriver',
            'driver=openmdao.core.driver:Driver',
            'simplegadriver=openmdao.drivers.genetic_algorithm_driver:SimpleGADriver',
            'differentialevolutiondriver=openmdao.drivers.differential_evolution_driver:DifferentialEvolutionDriver',
            'pyoptsparsedriver=openmdao.drivers.pyoptsparse_driver:pyOptSparseDriver',
            'scipydriver=openmdao.drivers.scipy_optimizer:ScipyOptimizeDriver',
        ],
        'openmdao_group': [
            'group=openmdao.core.group:Group',
            'parallelgroup=openmdao.core.parallel_group:ParallelGroup',
        ],
        'openmdao_lin_solver': [
            'directsolver=openmdao.solvers.linear.direct:DirectSolver',
            'linearblockgs=openmdao.solvers.linear.linear_block_gs:LinearBlockGS',
            'linearblockjac=openmdao.solvers.linear.linear_block_jac:LinearBlockJac',
            'linearrunoncec=openmdao.solvers.linear.linear_runonce:LinearRunOnce',
            'petsckrylov=openmdao.solvers.linear.petsc_ksp:PETScKrylov',
            'scipykrylov=openmdao.solvers.linear.scipy_iter_solver:ScipyKrylov',
            'userdefined=openmdao.solvers.linear.user_defined:LinearUserDefined',
        ],
        'openmdao_nl_solver': [
            'broydensolver=openmdao.solvers.nonlinear.broyden:BroydenSolver',
            'newtonsolver=openmdao.solvers.nonlinear.newton:NewtonSolver',
            'nonlinearblockgs=openmdao.solvers.nonlinear.nonlinear_block_gs:NonlinearBlockGS',
            'nonlinearblockjac=openmdao.solvers.nonlinear.nonlinear_block_jac:NonlinearBlockJac',
            'nonlinearrunonce=openmdao.solvers.nonlinear.nonlinear_runonce:NonlinearRunOnce',
            'armijogoldsteinls=openmdao.solvers.linesearch.backtracking:ArmijoGoldsteinLS',
            'boundsenforcels=openmdao.solvers.linesearch.backtracking:BoundsEnforceLS',
        ],
        'openmdao_report': [
            'n2=openmdao.visualization.n2_viewer.n2_viewer:_n2_report_register',
            'scaling=openmdao.visualization.scaling_viewer.scaling_report:_scaling_report_register',
            'optimizer=openmdao.visualization.opt_report.opt_report:_optimizer_report_register',
            'connections=openmdao.visualization.connection_viewer.viewconns:_connections_report_register',
            'inputs=openmdao.visualization.inputs_report.inputs_report:_inputs_report_register',
            'total_coloring=openmdao.utils.coloring:_total_coloring_report_register',
            'summary=openmdao.devtools.debug:_summary_report_register',
            'checks=openmdao.error_checking.check_config:_check_report_register',
        ],
        'openmdao_surrogate_model': [
            'krigingsurrogate=openmdao.surrogate_models.kriging:KrigingSurrogate',
            'nearestneighbor=openmdao.surrogate_models.nearest_neighbor:NearestNeighbor',
            'responsesurface=openmdao.surrogate_models.response_surface:ResponseSurface',
            'multificokrigingsurrogate=openmdao.surrogate_models.multifi_cokriging:MultiFiCoKrigingSurrogate',
        ]
    },
    extras_require=optional_dependencies,
)
