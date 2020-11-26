#!/usr/bin/env python

'''

'''
import json

from pyod import SGP4
from pyod import OptimizeLeastSquares, MCMCLeastSquares
from pyod import CameraStation

import pyod.plot as plot
from pyod.posterior import _named_to_enumerated

import pyorb

import numpy as np
import matplotlib.pyplot as plt

from alis4d_source import sources, prior_time, ecef_est

from sorts.population import tle_catalog
from sorts.frames import convert
from astropy.time import Time

steps = int(1e4)

prop = SGP4(
    settings=dict(
        in_frame='TEME',
        out_frame='ITRS',
    )
)

spacetrack = json.load(open('spacetrack.json','r'))
tles = [(x['TLE_LINE1'],x['TLE_LINE2']) for x in spacetrack]
pop = tle_catalog(tles, cartesian=False) #for SGP4
pop.filter('oid', lambda x: x == 39771)
obj = pop.get_object(0)
mean, B, epoch = obj.propagator.get_mean_elements(obj.state[0], obj.state[1])
mean[0] *= 1e-3
mean[2:] = np.radians(mean[2:])

print('Mean elements prior')
for key,val in zip(['a', 'e', 'i', 'raan', 'aop', 'mu'], mean.tolist()):
    print(f'{key:<4}: {val:.2f}')
exit()

dt = Time(prior_time) - epoch

obj.out_frame = 'TEME'
state0 = obj.get_state(dt)

params = dict(B=B)

variables = ['x', 'y', 'z', 'vx', 'vy', 'vz']
dtype = [(name, 'float64') for name in variables]

state0_named = np.empty((1,), dtype=dtype)
step_arr = np.array([1e3,1e3,1e3,1e1,1e1,1e1], dtype=np.float64)
step = np.empty((1,), dtype=dtype)

for ind, name in enumerate(variables):
    state0_named[name] = state0[ind]
    step[name] = step_arr[ind]


input_data_state = {
    'sources': sources,
    'Model': CameraStation,
    'date0': prior_time,
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

# post = post_init

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
    steps = steps,
    step = step,
    tune = 500,
)


post.run()
# post.results.save('SGP4_mcmc_results.h5')

mean_elements = prop.TEME_to_TLE(_named_to_enumerated(post.results.MAP, post.variables), Time(prior_time), B=B, kepler=False, tol=1e-5, tol_v=1e-7)

print('Mean elements')
for key,val in zip(['a', 'e', 'i', 'raan', 'aop', 'mu'], mean_elements.tolist()):
    print(f'{key:<4}: {val:.2f}')

print(post.results)


fig, axes = plt.subplots(3, 1,figsize=(15,15), sharex=True)

for label, __state, col in zip(['start','MAP'], [state0_named, post.results.MAP], ['b','k']):
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

for i in range(3):
    axes[i].plot(post._models[0].data['t']/3600, ecef_est[i,:],"-r",
        label='Triangulation',
    )

axes[0].legend()

plot.trace(post.results)
plot.scatter_trace(post.results)

# plot.orbits(post)
plot.residuals(post, [state0_named, post.results.MAP], ['Start', 'MAP'], ['-b', '-g'], absolute=False)
plot.residuals(post, [state0_named, post.results.MAP], ['Start', 'MAP'], ['-b', '-g'], absolute=True)

plt.show()


