ALIS4D SST
============

Using the orbit determination

.. code-block:: python

    #!/usr/bin/env python

    import matplotlib.pyplot as plt
    import alis4dsst as a4

    sources, time0, state0 = a4.io.load_track('Sat_coord_20200401T195000b.mat', 1, 1)
    fig, ax = a4.plots.track(sources)

    plt.show()


When used for publications
===========================

Contact at least one of the following:

 * Daniel Kastinen <daniel.kastinen@irf.se>
 * Tima Sergienko <tima@irf.se>
 * Urban Braendstroem <urban.brandstrom@irf.se>
 * Petrus Hyvönen <Petrus.Hyvonen@sscspace.com>
 * Johan Kero <kero@irf.se>