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
  get the latest version of Theano and Fuel. We recommand you select
  "Use Git from the Windows Command Prompt" option, so you can execute
  all the following command lines from the regular Windows `cmd` shell.

- Install a C++ compiler and Python DLL. From a shell:

  .. code-block:: winbatch

    conda install mingw libpython


Optional: Additional step to display the graphics
+++++++++++++++++++++++++++++++++++++++++++++++++
If you do not follow these steps, the `pydotprint` command will raise an exception and fail, but the other functionalities of Theano would still work.

On Ubuntu/Debian
~~~~~~~~~~~~~~~~

.. code-block:: bash

  $ sudo apt-get install graphviz
  $ conda install pydot

On Fedora, CentOS, Red Hat Enterprise
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: bash

  $ sudo yum install graphviz
  $ conda install pydot

On MacOS
~~~~~~~~
- Download graphviz from http://www.graphviz.org/Download_macos.php
- Then, from a terminal:

  .. code-block:: bash

    $ conda install pydot

On Windows
~~~~~~~~~~
- Download graphviz from http://www.graphviz.org/Download_windows.php
- Add to the `PATH` environment variable the directory where the
  binaries were installed, by default
  `C:\\Program Files (x86)\\Graphviz2.38\\bin`
- Then, from a terminal:

  .. code-block:: winbatch

    pip install pydot_ng

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

  $ pip install git+git://github.com/Theano/Theano.git --user

.. note::

  If you are using Windows and selected "Use Git from Git Bash only" when
  installing Git, or if the command above failed because git is not
  available in the path, then you need to run the command line above
  from the "Git Bash" terminal instead of the regular Windows command
  prompt.

If you are following these instructions in advance, you may need to
execute this command in order to get last-minute fixes:

.. code-block:: bash

  $ pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git --user

.. note::

  If you install Theano for the current user only (using `--user`),
  command-line utilities (for instance `theano-cache`) will not be
  accessible from a terminal directly. You would have to add the script
  installation directory to the `PATH` environment variable.

  - On Mac OS and Linux, that path is `$HOME/.local/bin` by default.
  - On Windows 7, that path is `C:\\<User>\\AppData\\Roaming\\Python\\Scripts`
    if your user name is "<User>".


Fuel
----

We install the development version of Fuel from GitHub.

.. code-block:: bash

  $ pip install git+git://github.com/mila-udem/fuel.git --user

If you are following these instructions in advance, you may need to
execute this command in order to get last-minute fixes:

.. code-block:: bash

  $ pip install --upgrade --no-deps git+git://github.com/mila-udem/fuel.git --user

.. note::

  If you install Fuel for the current user only (using `--user`),
  command-line utilities (for instance `fuel-download` and `fuel-convert`)
  will not be accessible from a terminal directly. Unless you have already
  performed that step when installing Theano, you would have to add the script
  installation directory to the `PATH` environment variable.

  - On Mac OS and Linux, that path is `$HOME/.local/bin` by default.
  - On Windows 7, that path is `C:\\<User>\\AppData\\Roaming\\Python\\Scripts`
    if your user name is "<User>".


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
