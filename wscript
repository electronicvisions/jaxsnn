#!/usr/bin/env python
from waflib.extras.test_base import summary
import copy
import site
import os
import re
import sys
from waflib.extras.symwaf2ic import get_toplevel_path


def depends(dep):
    dep('code-format')


def options(opt):
    opt.load('test_base')
    opt.load('python')
    opt.load('pytest')
    opt.load('pylint')
    opt.load('pycodestyle')
    opt.load('doxygen')


def configure(cfg):
    cfg.load('test_base')
    cfg.load('python')
    cfg.check_python_version()
    cfg.check_python_headers()
    cfg.load('pytest')
    cfg.load('doxygen')


def build(bld):
    bld(
        target='jaxsnn',
        features='py use',
        #use=[],
        relative_trick=True,
        source=bld.path.ant_glob('src/pyjaxsnn/**/*.py'),
        install_path = '${PREFIX}/lib',
        install_from='src/pyjaxsnn',
    )

    bld(
        target='jaxsnn_linting',
        features='py use pylint pycodestyle',
        #use=[],
        relative_trick=True,
        source=bld.path.ant_glob('src/pyjaxsnn/**/*.py'),
        pylint_config=os.path.join(get_toplevel_path(), "code-format", "pylintrc"),
        pycodestyle_config=os.path.join(get_toplevel_path(), "code-format", "pycodestyle")
    )

    bld(
        target='jaxsnn_swtests',
        tests=bld.path.ant_glob('tests/sw/*.py'),
        features='use pytest',
        use=['jaxsnn'],
        install_path='${PREFIX}/bin/tests/sw',
    )

    bld(
        target = 'doxygen_pyjaxsnn',
        features = 'doxygen',
        doxyfile = bld.root.make_node(os.path.join(get_toplevel_path(), "code-format", "doxyfile")),
        doxy_inputs = 'src/pyjaxsnn',
        install_path = 'doc/pycwjaxsnnhxtorch',
        pars = {
            "PROJECT_NAME": "\"pyjaxsnn\"",
            "OUTPUT_DIRECTORY": os.path.join(get_toplevel_path(), "build", "pyjaxsnn", "doc"),
            "PYTHON_DOCSTRING": "NO",
        },
    )


# Create test summary (to stdout and XML file)
    bld.add_post_fun(summary)
