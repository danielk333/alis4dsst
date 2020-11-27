#!/usr/bin/env python

'''

'''
import logging

import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time

import sorts
import pyorb
from pyod import SGP4
from pyod import OptimizeLeastSquares, MCMCLeastSquares
from pyod import CameraStation

from . import alis4d


def get_mean_elements_start(tle_space_object, start_epoch):
    '''
    '''

    mean0, B, tle_epoch = tle_space_object.propagator.get_mean_elements(tle_space_object.state[0], tle_space_object.state[1], radians=False)

    params0 = dict(B=B)

    if start_epoch is None:
        epoch0 = tle_epoch
        return mean0, params0, epoch0

    elif isinstance(start_epoch, Time):
        epoch0 = start_epoch
    else:
        epoch0 = Time(start_epoch, format='datetime64', scale='utc')

    dt = (epoch0 - tle_epoch).sec

    out_frame_ = tle_space_object.out_frame
    tle_space_object.out_frame = 'TEME'

    teme0 = tle_space_object.get_state(dt)

    tle_space_object.out_frame = out_frame_

    mean_start = tle_space_object.propagator.TEME_to_TLE(teme0[:,0], epoch0, B=B, kepler=False)

    mean0[5] = mean_start[5]

    return mean0, params0, epoch0



def determine_orbit(sources, start, propagator, epoch, mcmc=False, **kwargs):

    ret = dict()

    samples = kwargs.get('samples', int(1e5))
    step_arr = kwargs.get('step', np.array([1e3,1e-1,1.,1.,1.,1.], dtype=np.float64))

    logger = sorts.profiling.get_logger('od', term_level = logging.ERROR)

    if propagator.lower() == 'orekit':
        from sorts.propagator import Orekit

        prop = Orekit(
            orekit_data = kwargs['orekit_data'], 
            settings=dict(
                in_frame='TEME',
                out_frame='ITRS',
                drag_force = kwargs.get('drag_force',False),
                radiation_pressure = False,
            ),
            logger = logger,
        )
        params = dict()
        variables = ['x', 'y', 'z', 'vx', 'vy', 'vz']

    elif propagator.lower() == 'mean-elements':
        prop = SGP4(
            settings=dict(
                in_frame='TEME',
                out_frame='ITRS',
            ),
            logger = logger,
        )
        params = dict(SGP4_mean_elements=True)
        variables = ['a', 'e', 'i', 'raan', 'aop', 'mu']

    elif propagator.lower() == 'sgp4':
        prop = SGP4(
            settings=dict(
                in_frame='TEME',
                out_frame='ITRS',
            ),
            logger = logger,
        )
        params = dict(SGP4_mean_elements=False)
        variables = ['x', 'y', 'z', 'vx', 'vy', 'vz']
    else:
        raise ValueError(f'Propagator "{propagator}" not recognized.')

    params.update(kwargs.get('params', {}))
    dtype = [(name, 'float64') for name in variables]

    if isinstance(epoch, Time):
        epoch0 = epoch
    else:
        epoch0 = Time(epoch, format='datetime64', scale='utc')

    state0_named = np.empty((1,), dtype=dtype)
    step = np.empty((1,), dtype=dtype)

    for ind, name in enumerate(variables):
        state0_named[name] = start[ind]
        step[name] = step_arr[ind]

    ret['state0_named'] = state0_named
    ret['step'] = step
    ret['variables'] = variables

    input_data_state = {
        'sources': sources,
        'Model': CameraStation,
        'date0': epoch0.datetime64,
        'params': params,
    }

    post_init = OptimizeLeastSquares(
        data = input_data_state,
        variables = variables,
        start = state0_named,
        prior = None,
        propagator = prop,
        method = 'Nelder-Mead',
        options = dict(
            maxiter = 10000,
            disp = False,
            xatol = 1e-3,
        ),
    )

    post_init.run()

    ret['post_init'] = post_init

    if mcmc:
        post = MCMCLeastSquares(
            data = input_data_state,
            variables = variables,
            start = post_init.results.MAP,
            prior = None,
            propagator = prop,
            method = 'SCAM',
            method_options = dict(
                accept_max = 0.5,
                accept_min = 0.3,
                adapt_interval = 500,
            ),
            steps = samples,
            step = step,
            tune = 500,
        )


        post.run()
    else:
        post = post_init

    ret['post'] = post

    return ret




