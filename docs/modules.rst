Modules documentation
=====================

Filters:
--------

Triple Moving Average
~~~~~~~~~~~~~~~~~~~~~
.. automodule:: orbitdeterminator.filters.triple_moving_average
   :members:

Savintzky - Golay
~~~~~~~~~~~~~~~~~
.. automodule:: orbitdeterminator.filters.sav_golay
   :members:

Interpolation:
--------------

Lamberts-Kalman Method
~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: orbitdeterminator.kep_determination.lamberts_kalman
   :members:

Gibb's Method
~~~~~~~~~~~~~
.. autoclass:: orbitdeterminator.kep_determination.gibbsMethod.Gibbs
   :members:

Spline Interpolation
~~~~~~~~~~~~~~~~~~~~
.. automodule:: orbitdeterminator.kep_determination.interpolation
   :members:

Ellipse Fit
~~~~~~~~~~~~~~~~~~~~
.. automodule:: orbitdeterminator.kep_determination.ellipse_fit
   :members:

Gauss method
~~~~~~~~~~~~~~~~~~~~
.. automodule:: orbitdeterminator.kep_determination.gauss_method
   :members:

Least squares
~~~~~~~~~~~~~~~~~~~~
.. automodule:: orbitdeterminator.kep_determination.least_squares
   :members:

Propagation:
------------

Propagation Model
~~~~~~~~~~~~~~~~~
.. autoclass:: orbitdeterminator.propagation.sgp4.SGP4
   :special-members: __init__
   :members:

.. autoclass:: orbitdeterminator.propagation.sgp4.FlagCheckError
   :members:

Cowell Method
~~~~~~~~~~~~~~~~~
.. automodule:: orbitdeterminator.propagation.cowell
   :members:

Simulator
~~~~~~~~~~~~~~~~~
.. autoclass:: orbitdeterminator.propagation.simulator.Simulator
   :special-members: __init__
   :members:

.. autoclass:: orbitdeterminator.propagation.simulator.SimParams
   :members:

.. autoclass:: orbitdeterminator.propagation.simulator.OpWriter
   :members:

.. autoclass:: orbitdeterminator.propagation.simulator.print_r
   :show-inheritance:

.. autoclass:: orbitdeterminator.propagation.simulator.save_r
   :show-inheritance:
   :members: __init__


DGSN Simulator
~~~~~~~~~~~~~~~~~
.. autoclass:: orbitdeterminator.propagation.dgsn_simulator.DGSNSimulator
   :special-members: __init__
   :members:

.. autoclass:: orbitdeterminator.propagation.dgsn_simulator.SimParams
   :members:

.. autoclass:: orbitdeterminator.propagation.dgsn_simulator.OpWriter
   :members:

.. autoclass:: orbitdeterminator.propagation.dgsn_simulator.print_r
   :show-inheritance:

.. autoclass:: orbitdeterminator.propagation.dgsn_simulator.save_r
   :show-inheritance:
   :members: __init__

Kalman Filter
~~~~~~~~~~~~~~~~~
.. automodule:: orbitdeterminator.propagation.kalman_filter
   :members:

sgp4_prop
~~~~~~~~~~~~~~~~~
.. automodule:: orbitdeterminator.propagation.sgp4_prop
   :members:

sgp4_prop_string
~~~~~~~~~~~~~~~~~
.. automodule:: orbitdeterminator.propagation.sgp4_prop_string
   :members:

Utils:
------

kep_state
~~~~~~~~~
.. automodule:: orbitdeterminator.util.kep_state
   :members:

read_data
~~~~~~~~~
.. automodule:: orbitdeterminator.util.read_data
   :members:

state_kep
~~~~~~~~~
.. automodule:: orbitdeterminator.util.state_kep
   :members:

input_transf
~~~~~~~~~~~~
.. automodule:: orbitdeterminator.util.input_transf
   :members:

rkf78
~~~~~
.. automodule:: orbitdeterminator.util.rkf78
   :members:

golay_window
~~~~~~~~~~~~
.. automodule:: orbitdeterminator.util.golay_window
   :members:

anom_conv
~~~~~~~~~~~~
.. automodule:: orbitdeterminator.util.anom_conv
   :members:

new_tle_kep_state
~~~~~~~~~~~~~~~~~
.. automodule:: orbitdeterminator.util.new_tle_kep_state
   :members:

teme_to_ecef
~~~~~~~~~~~~
.. automodule:: orbitdeterminator.util.teme_to_ecef
   :members:
