==========
my_project
==========

This is the documentation of **my_project**.

.. note::

    This is the main page of your project's `Sphinx <http://sphinx-doc.org/>`_
    documentation. It is formatted in `reStructuredText
    <http://sphinx-doc.org/rest.html>`__. Add additional pages by creating
    rst-files in ``docs`` and adding them to the `toctree
    <http://sphinx-doc.org/markup/toctree.html>`_ below. Use then
    `references <http://sphinx-doc.org/markup/inline.html>`__ in order to link
    them from this page, e.g. :ref:`authors <authors>` and :ref:`changes`.

    It is also possible to refer to the documentation of other Python packages
    with the `Python domain syntax
    <http://sphinx-doc.org/domains.html#the-python-domain>`__. By default you
    can reference the documentation of `Sphinx <http://sphinx.pocoo.org>`__,
    `Python <http://docs.python.org/>`__, `NumPy
    <http://docs.scipy.org/doc/numpy>`__, `SciPy
    <http://docs.scipy.org/doc/scipy/reference/>`__, `matplotlib
    <http://matplotlib.sourceforge.net>`__, `Pandas
    <http://pandas.pydata.org/pandas-docs/stable>`__, `Scikit-Learn
    <http://scikit-learn.org/stable>`__. You can add more by
    extending the ``intersphinx_mapping`` in your Sphinx's ``conf.py``.

    The pretty useful extension `autodoc
    <http://www.sphinx-doc.org/en/stable/ext/autodoc.html>`__ is activated by
    default and lets you include documentation from docstrings. Docstrings can
    be written in `Google
    <http://google.github.io/styleguide/pyguide.html#Comments>`__
    (recommended!), `NumPy
    <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`__
    and `classical
    <http://www.sphinx-doc.org/en/stable/domains.html#info-field-lists>`__
    style.


Contents
========

.. toctree::
   :maxdepth: 2

   License <license>
   Authors <authors>
   Changelog <changes>
   Module Reference <api/modules>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
