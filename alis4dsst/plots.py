from pyod.plot import earth_grid
import sorts
from pyod.posterior import _named_to_enumerated

import matplotlib.pyplot as plt
import numpy as np
from astropy.time import TimeDelta

from . import alis4d

def track(S):

    fig, axes = plt.subplots(2, 1, sharex=True)

    for i in range(len(S)):
        az = S[i].data['az']
        el = S[i].data['el']
        t = S[i].data['date']
        t = (t - min(t))/np.timedelta64(60, 's')

        axes[0].plot(t, az, label=alis4d.stations[S[i].kwargs['station']]['name'])
        axes[1].plot(t, el)
        axes[1].set_xlabel('Time relative track start [min]')
        axes[0].set_ylabel('Azimuth [deg]')
        axes[1].set_ylabel('Elevation [deg]')

    axes[0].legend()
    
    return fig, axes


def ang_mod(theta):
    return np.mod(theta + 360.0, 360.0)

def model_to_data(start, post, ecef_ref=None, ref_label='Triangularization', axes=None):

    if axes is None:
        fig, axes = plt.subplots(3, 1,figsize=(15,15), sharex=True)
    else:
        fig = None

    for label, __state, col in zip(['start','MAP'], [start, post.results.MAP], ['b','k']):
        __state = _named_to_enumerated(__state, post.variables)
        t = np.linspace(0, np.max(post._models[0].data['t']), num=1000)
        _t = post._models[0].data['t']
        post._models[0].data['t'] = t
        states_ = post._models[0].get_states(__state)
        post._models[0].data['t'] = _t

        for i in range(3):
            axes[i].plot(t/3600, states_[i,:],"-"+col,
                label=label,
            )

    if ecef_ref is not None:
        for i in range(3):
            axes[i].plot(post._models[0].data['t']/3600, ecef_ref[i,:],"-r",
                label=ref_label,
            )

    axes[0].legend()

    return fig, axes


def correlation_azel(dat, cdat, station=None, axes=None):
    '''Plot the correlation between the measurement and simulated population object.
    '''
    t = dat['t']
    epoch = dat['epoch']
    az = ang_mod(dat['az'])
    el = dat['el']
    az_ref = ang_mod(cdat['az_ref'])
    el_ref = cdat['el_ref']

    azx = np.cos(np.radians(az))
    azy = np.sin(np.radians(az))

    azx_ref = np.cos(np.radians(az_ref))
    azy_ref = np.sin(np.radians(az_ref))

    ang = np.degrees(np.arccos(azx*azx_ref + azy*azy_ref))

    if axes is None:
        fig = plt.figure()
        axes = [fig.add_subplot(211), fig.add_subplot(212)]
    else:
        fig = None

    ax = axes[0]
    ax.plot(t - t[0], az, label='measurement')
    ax.plot(t - t[0], az_ref, label='simulation')
    ax.set_ylabel('Azimuth [deg]')
    ax.set_xlabel('Time [s]')

    ax = axes[1]
    ax.plot(t - t[0], el, label='measurement')
    ax.plot(t - t[0], el_ref, label='simulation')
    ax.set_ylabel('Elevation [deg]')
    ax.set_xlabel('Time [s]')

    if station is not None:
        fig.suptitle(alis4d.stations[station]['name'])

    return fig, axes


def correlation_resid(dat, cdat, station=None, axes=None):
    '''Plot the correlation between the measurement and simulated population object.
    '''
    t = dat['t']
    epoch = dat['epoch']
    az = ang_mod(dat['az'])
    el = dat['el']
    az_ref = ang_mod(cdat['az_ref'])
    el_ref = cdat['el_ref']

    azx = np.cos(np.radians(az))
    azy = np.sin(np.radians(az))

    azx_ref = np.cos(np.radians(az_ref))
    azy_ref = np.sin(np.radians(az_ref))

    ang = np.degrees(np.arccos(azx*azx_ref + azy*azy_ref))

    if axes is None:
        fig = plt.figure()
        axes = [fig.add_subplot(221 + i) for i in range(4)]
    else:
        fig = None


    ax = axes[0]
    if ax is not None:
        ax.hist(ang)
        ax.set_xlabel('Azimuth residuals [deg]')

    ax = axes[1]
    if ax is not None:
        ax.hist(el_ref - el)
        ax.set_xlabel('Elevation residuals [deg]')

    
    ax = axes[2]
    if ax is not None:
        ax.plot(t - t[0], ang)
        ax.set_ylabel('Azimuth residuals [deg]')
        ax.set_xlabel('Time [s]')


    ax = axes[3]
    if ax is not None:
        ax.plot(t - t[0], el_ref - el)
        ax.set_ylabel('Elevation residuals [deg]')
        ax.set_xlabel('Time [s]')


    if station is not None:
        fig.suptitle(alis4d.stations[station]['name'])

    return fig, axes


def correlation_track(dat, cdat, station=None, ax=None):
    '''Plot the correlation between the measurement and simulated population object.
    '''
    t = dat['t']
    epoch = dat['epoch']
    az = ang_mod(dat['az'])
    el = dat['el']
    az_ref = ang_mod(cdat['az_ref'])
    el_ref = cdat['el_ref']

    azx = np.cos(np.radians(az))
    azy = np.sin(np.radians(az))

    azx_ref = np.cos(np.radians(az_ref))
    azy_ref = np.sin(np.radians(az_ref))

    ang = np.degrees(np.arccos(azx*azx_ref + azy*azy_ref))

    if ax is None:
        fig, ax = plt.subplots(1,1)
    else:
        fig = None

    fig_, ax = sorts.plotting.local_tracking(
        cdat['az_ref'], 
        cdat['el_ref'], 
        ax=ax, 
        t=epoch + TimeDelta(t, format='sec'),
    )
    fig_, ax = sorts.plotting.local_tracking(
        dat['az'], 
        dat['el'], 
        ax=ax, 
        t=epoch + TimeDelta(t, format='sec'),
        add_track = True,
    )

    if station is not None:
        fig.suptitle(alis4d.stations[station]['name'])

    return fig, ax