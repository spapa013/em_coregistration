#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='em2p_coreg',
    version='0.0.1',
    description='Coregistration code',
    author='Dan Kapner',
    author_email='danielk@alleninstitute.org',
    packages=find_packages(exclude=[]),
    install_requires=['numpy', 'tqdm', 'scipy', 'pandas', 'datajoint', 'meshparty', 'pykdtree', 'ipyvolume', 'matplotlib', 'argschema', 'torch']
)
