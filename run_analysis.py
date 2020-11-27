
import json

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from pyod import PosteriorParameters
from sorts.population import tle_catalog
import pyod.plot as odplot

import alis4dsst as a4

def run_correlator(fname, json_file):
    spacetrack = json.load(open(json_file,'r'))

    sources, time0, state0 = a4.io.load_track(fname, 1, 1)

    tles = [(x['TLE_LINE1'],x['TLE_LINE2']) for x in spacetrack]

    measurements, indecies, metric, cdat = a4.correlate(
        sources, 
        tles, 
        propagator='SGP4',
        oids = [39771],
    )

    fig, axes = plt.subplots(4,len(sources), sharex=True)
    axes[0,0].legend()
    for i in range(len(sources)):
        axes[0,i].set_title(a4.alis4d.stations[sources[i].kwargs['station']]['name'])
        a4.plots.correlation_azel(measurements[i], cdat[0][i], axes = [axes[j,i] for j in range(2)])
        a4.plots.correlation_resid(measurements[i], cdat[0][i], axes = [None]*2 + [axes[j,i] for j in range(2,4)])

    a4.plots.correlation_track(measurements[0], cdat[0][0])

    plt.show()
