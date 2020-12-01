from pyod import SourcePath
from pyod.coordinates import geodetic2ecef

import numpy as np
import scipy.io as sio

from . import alis4d

def load_sds(fname, n_est=int(1e6)):
    '''Load the PSF and estimate standard deviations from Tima's matlab file.
    '''

    mat_err = sio.loadmat(fname)
    mat_err['amb_fun'] = mat_err['amb_fun'].flatten()/mat_err['amb_fun'].flatten().sum()
    cumsum = np.cumsum(mat_err['amb_fun'])
    mat_err['az_amb'] = np.degrees(mat_err['az_amb'].flatten())
    mat_err['ze_amb'] = np.degrees(mat_err['ze_amb'].flatten())

    np.random.seed(1234) #for consistency
    rng_ = np.random.rand(n_est)
    np.random.seed(None)

    rng_inds = np.zeros(rng_.shape, dtype=np.int64)

    for i in range(n_est):
        rng_inds[i] = np.argmax(rng_[i] < cumsum)

    az_samps = mat_err['az_amb'][rng_inds]
    el_samps = 90.0 - mat_err['ze_amb'][rng_inds]

    az_sd = np.std(az_samps)
    el_sd = np.std(el_samps)

    return az_sd, el_sd, az_samps, el_samps

def load_track(fname, az_sd, el_sd):
    '''Load Satellite track data from Tima's matlab file format and create `pyod` input data instances.
    '''

    mat = sio.loadmat(fname)

    file_stations = [name for name in mat if name.startswith('az')]
    file_stations = [name[2] for name in file_stations if len(name) > 2]
    file_stations = set(file_stations)

    kwargs = dict(
        path = SourcePath(fname, 'file'),
        az_sd = az_sd,
        el_sd = el_sd,
    )

    sources = [
        alis4d.ALID4DTrack(
            station = name,
            ecef = alis4d.stations[name]['ecef'],
            **kwargs
        )
        for name in file_stations
    ]

    time0 = sources[0].data['date'][0]

    #if triangulation is available, create a OD start value
    if 'latSat' in mat:
        prior_ecef_start = geodetic2ecef(mat['latSat'][0,0], mat['longSat'][0,0], mat['altSat'][0,0]*1e3, radians=False)

        prior_vel = np.zeros((3,))
        for j in range(len(sources[0].data['date'])-1):
            prior_ecef0 = geodetic2ecef(mat['latSat'][j,0], mat['longSat'][j,0], mat['altSat'][j,0]*1e3, radians=False)
            prior_ecef1 = geodetic2ecef(mat['latSat'][j+1,0], mat['longSat'][j+1,0], mat['altSat'][j+1,0]*1e3, radians=False)

            time_delta = (sources[0].data['date'][j+1] - sources[0].data['date'][j])/np.timedelta64(1, 's')
            prior_vel += (prior_ecef1 - prior_ecef0)/time_delta

        prior_vel /= len(sources[0].data['date'])-1
        
        state0 = np.empty((6,))
        state0[:3] = prior_ecef_start
        state0[3:] = prior_vel
    else:
        state0 = None

    return sources, time0, state0