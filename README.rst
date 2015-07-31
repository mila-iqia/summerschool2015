summerschool2015
================

This repository contains the slides and exercises for the `Deep Learning
Summer School 2015`_ programming tutorials.


Installation instructions
=========================

The tutorials are written in Python, using Theano_ and Fuel_. They are
designed to be run locally on a laptop, without using a GPU.


Python and dependencies
-----------------------

The simplest way to install a Python software stack with most
dependencies is to use Anaconda_.

First, download and execute the installer. You can install it as a user
(you do not have to use `sudo`). We recommend that you let the installer
make Anaconda the default Python version.

Then, in a terminal:

.. code-block:: bash

  $ conda update conda


Additional steps for Windows
++++++++++++++++++++++++++++

These additional steps are required for Windows:

- Download Git_, and execute the installer. This will be necessary to
  get the latest version of Theano and Fuel.

- Install a C++ compiler and Python DLL. From a shell:

  .. code-block:: bash

    conda install mingw libpython


Opt: Additional step to display the graphics
++++++++++++++++++++++++++++++++++++++++++++

  $ conda install pydot

Under Ubuntu/Debian:

  $ sudo apt-get install graphviz

Under Fedora, Cent OS, Red Hat Enterprise:

  $ sudo yum install graphviz

Under Mac:

  Download graphviz from: http://www.graphviz.org/Download_macos.php

Under Windows:

  Download graphvix from: http://www.graphviz.org/Download_windows.php

Optional: MKL Blas
++++++++++++++++++

If you are eligible to an `academic license`_ for Anaconda add-ons, you
can download and install the `MKL optimizations`_. This will bring a
small speed improvement for dot products, but is not critical for the
tutorials at all. Once you have obtained the license:

.. code-block:: bash

  $ conda install mkl


Theano
------

There have been some improvement and bug fixes since the last release,
so we will use the latest development version from GitHub. The following
command installs it for the current user only:

.. code-block:: bash

  $ pip install --upgrade git+git://github.com/Theano/Theano.git --user

If you are following these instructions in advance, you may need to
execute that command again in order to get last-minute fixes.


Fuel
----

Some of Fuel's dependencies have to be installed via `conda`.
Then, we install the development version of Fuel from GitHub.

.. code-block:: bash

  $ conda install pillow
  $ pip install --upgrade git+git://github.com/mila-udem/fuel.git --user


Get and run these tutorials
===========================

First, clone this repository:

.. code-block:: bash

  $ git clone https://github.com/mila-udem/summerschool2015.git

To use the IPython notebooks, you have to launch the IPython server on the
base directory:

.. code-block:: bash

  $ ipython notebook summerschool2015

A new window or tab should open in your web browser. If it does not (or if you
want to use it in a different browser), the previous command should mention a
URL you can open, probably `<http://localhost:8888/>`__. From there, you can
navigate to the `.ipynb` files.


.. _Deep Learning Summer School 2015: https://sites.google.com/site/deeplearningsummerschool/
.. _Anaconda: http://continuum.io/downloads
.. _academic license: https://store.continuum.io/cshop/academicanaconda
.. _MKL optimizations: https://store.continuum.io/cshop/mkl-optimizations/
.. _Git: https://git-scm.com/download/win
.. _Theano: http://deeplearning.net/software/theano/
.. _Fuel: https://fuel.readthedocs.org/
