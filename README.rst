spmlib
======
Python tools for handling data from Dimitris Menemenlis' 1/48 degree global numerical ocean simulation for the Samoan Passage region (Box 12).

Modules
-------

* data: Tools for reading binary model data files into an `xarray  <http://xarray.pydata.org/en/stable/>`_ `Dataset <http://xarray.pydata.org/en/stable/data-structures.html#dataset>`_.

* plt: Tools for plotting and extracting data (latter should be moved to its own submodule I guess)

Installation
------------

First clone or download the repository. Then install using pip:

``cd spmlib``

``pip install .``

Optionally, use the the -e flag, to make it editable.

``pip install -e .``