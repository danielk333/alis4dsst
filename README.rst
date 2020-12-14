ALIS4D SST
============

Further work:
* implement proper PSF to use in logPost
* implement new posterior in pyod that uses PSF in posterior calculation

Running the analysis from terminal

.. code-block:: bash

    python run_analysis.py data/Sat_coord_20200401T195300.mat corr plot

Just add on more keywords at the end for options, they are

* corr: run correlation with spacetrack catalogue
* od: orbit determination using minimization
* mcmc: orbit determination using Markov Chain Monte Carlo
* plot: generate plots
* sgp4: use sgp4 mean elements for orbit determination
* orekit: use orekit for orbit determination 
* sgp4-state: use sgp4 but with a TEME state for orbit determination (default)
* override: override the caches if they exist
* forward: do a forward propagation of the MCMC results (needs mcmc to have completed)



Using the code in python

.. code-block:: python

    #!/usr/bin/env python

    import matplotlib.pyplot as plt
    import alis4dsst as a4

    sources, time0, state0 = a4.io.load_track('Sat_coord_20200401T195000b.mat', 1, 1)
    fig, ax = a4.plots.track(sources)

    plt.show()


Example runs

.. code-block:: bash

    python run_analysis.py data/Sat_coord_20200401T195000b.mat corr plot
    python run_analysis.py data/Sat_coord_20200401T195000b.mat od sgp4-state mcmc forward plot



When used for publications
===========================

Contact at least one of the following:

 * Daniel Kastinen <daniel.kastinen@irf.se>
 * Tima Sergienko <tima@irf.se>
 * Urban Braendstroem <urban.brandstrom@irf.se>
 * Petrus Hyvönen <Petrus.Hyvonen@sscspace.com>
 * Johan Kero <kero@irf.se>
