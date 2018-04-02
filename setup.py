"""spmlib setup
"""

from setuptools import find_packages, setup

# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='spmlib',
      version='0.1.0',
      description='python tools for reading binary mitgcm model output for Samoan Passage',
      long_description=long_description,
      author='Gunnar Voet',
      author_email='gvoet@ucsd.edu',
      license='MIT',
      url='https://github.com/gunnarvoet/spmlib',
      install_requires=['numpy', 'pandas', 'xarray', 'dask'],
      packages=find_packages(),
      )