import json

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from pyod import PosteriorParameters
import alis4dsst as a4


def plot_errors(fname):
    az_sd, el_sd, az_samps, el_samps = a4.io.load_sds(fname)

    fig, axes = plt.subplots(1, 2)
    axes[0].hist(az_samps)
    axes[1].hist(el_samps)
    
    mat_err = sio.loadmat(fname)

    fig, ax = plt.subplots(1, 1)
    ax.pcolor(np.degrees(mat_err['az_amb']), np.degrees(90 - mat_err['ze_amb']), mat_err['amb_fun'])

    plt.show()


def plot_file_raw_data(fname):

    sources, time0, state0 = a4.io.load_track(fname, 1, 1)
    fig, ax = a4.plots.track(sources)

    plt.show()

def test_correlate(fname, json_file):

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



if __name__=='__main__':
    #plot_file_raw_data('data/Sat_coord_20200401T195000b.mat')
    #test_correlate('data/Sat_coord_20200401T195000b.mat', 'data/spacetrack.json')
    plot_errors('data/amb_function.mat')