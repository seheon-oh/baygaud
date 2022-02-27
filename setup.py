#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import re
import glob
try:
    from setuptools import setup
    setup
except ImportError:
    from distutils.core import setup
    setup

dir_path = os.path.dirname(os.path.realpath(__file__))

init_string = open(os.path.join(dir_path, 'src', 'baygaud', '__init__.py')).read()
VERS = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VERS, init_string, re.M)
__version__ = mo.group(1)

try:
    import pypandoc
    with open('README.md', 'r') as f:
        txt = f.read()
    txt = re.sub('<[^<]+>', '', txt)
    long_description = pypandoc.convert(txt, 'rst', 'md')
except ImportError:
    long_description = open('README.md').read()

 
setup(
    name                = 'baygaud',
    version             = '0.1.9',
    description         = 'profile decomposition tool',
    author              = 'Se-Heon Oh',
    author_email        = 'seheon.oh@sejong.ac.kr',
    url                 = 'https://github.com/seheon.oh/baygaud',
    download_url        = 'https://github.com/seheon.oh/baygaud/xx.zip',
    install_requires    =  ['numpy', 'fitsio', 'astropy', 'spectral_cube', 'ray', 'dynesty', 'numba'],
    packages            = ["baygaud"],
    keywords            = ['HI spectral line decomposition', 'nested sampling'],
    python_requires     = '>=3',
    package_data        = {"": ["README.md"]},
    package_dir			= {'': 'src/'},
    include_package_data=True,
    zip_safe            = False,
    classifiers         = [
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
