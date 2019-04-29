import re

from setuptools import setup


__version__ = re.findall(
    r"""__version__ = ["']+([0-9\.]*)["']+""",
    open('openmdao/__init__.py').read(),
)[0]


optional_dependencies = {
    'docs': [
        'matplotlib',
        'mock',
        'numpydoc',
        'redbaron',
        'sphinx',
    ],
    'test': [
        'coverage',
        'parameterized',
        'numpydoc',
        'pycodestyle==2.3.1',
        'pydocstyle==2.0.0',
        'testflo>=1.3.4',
    ],
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
    description="OpenMDAO v2 framework infrastructure",
    long_description="""OpenMDAO is an open-source high-performance computing platform
    for systems analysis and multidisciplinary optimization, written in Python. It
    enables you to decompose your models, making them easier to build and maintain,
    while still solving them in a tightly coupled manner with efficient parallel numerical methods.
    """,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
    ],
    keywords='optimization multidisciplinary multi-disciplinary analysis',
    author='OpenMDAO Team',
    author_email='openmdao@openmdao.org',
    url='http://openmdao.org',
    download_url='http://github.com/OpenMDAO/OpenMDAO/tarball/'+__version__,
    license='Apache License, Version 2.0',
    packages=[
        'openmdao',
        'openmdao.approximation_schemes',
        'openmdao.code_review',
        'openmdao.components',
        'openmdao.core',
        'openmdao.devtools',
        'openmdao.devtools.problem_viewer',
        'openmdao.devtools.iprofile_app',
        'openmdao.devtools.xdsm_viewer',
        'openmdao.docs',
        'openmdao.docs._exts',
        'openmdao.docs._utils',
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
        'openmdao.test_suite',
        'openmdao.test_suite.components',
        'openmdao.test_suite.groups',
        'openmdao.test_suite.test_examples',
        'openmdao.test_suite.test_examples.beam_optimization',
        'openmdao.test_suite.test_examples.beam_optimization.components',
        'openmdao.utils',
        'openmdao.vectors',
        'openmdao.surrogate_models',
        'openmdao.surrogate_models.nn_interpolators'
    ],
    package_data={
        'openmdao.devtools': ['*.wpr', '*.html'],
        'openmdao.devtools.problem_viewer': [
            'visualization/libs/*.js',
            'visualization/src/*.js',
            'visualization/style/*.css',
            'visualization/style/*.woff',
            'visualization/*.html'
        ],
        'openmdao.devtools.xdsm_viewer': [
            'XDSMjs/*',
            'XDSMjs/src/*.js',
            'XDSMjs/build/*.js',
            'XDSMjs/test/*.js',
            'XDSMjs/test/*.html',
            'XDSMjs/examples/*.json',
        ],
        'openmdao.devtools.iprofile_app': [
            'static/*.html',
            'templates/*.html'
        ],
        'openmdao.docs': ['*.py', '_utils/*.py'],
        'openmdao.recorders': ['tests/legacy_sql/*.sql'],
        'openmdao.utils': ['unit_library.ini'],
        'openmdao.test_suite': [
            '*.py',
            '*/*.py',
            'matrices/*.npz'
        ],
        'openmdao': ['*/tests/*.py', '*/*/tests/*.py', '*/*/*/tests/*.py']
    },
    install_requires=[
        'networkx>=2.0',
        'numpy',
        'pyDOE2',
        'pyparsing',
        'scipy',
        'six',
    ],
    # scripts=['bin/om-pylint.sh']
    entry_points="""
    [console_scripts]
    wingproj=openmdao.devtools.wingproj:run_wing
    webview=openmdao.devtools.webview:webview_argv
    run_test=openmdao.devtools.run_test:run_test
    openmdao=openmdao.utils.om:openmdao_cmd
    """,
    extras_require=optional_dependencies,
)
