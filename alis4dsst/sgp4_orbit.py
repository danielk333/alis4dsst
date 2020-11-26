#!/usr/bin/env python

'''

'''
import json

from pyod import SGP4
from pyod import OptimizeLeastSquares, MCMCLeastSquares
from pyod import CameraStation

import pyod.plot as plot

import pyorb

import numpy as np
import matplotlib.pyplot as plt

from alis4d_source import sources, state0, prior_time, ecef_est

from sorts.population import tle_catalog
from sorts.frames import convert
from astropy.time import Time

steps = int(1e5)

prop = SGP4(
    settings=dict(
        in_frame='TEME',
        out_frame='ITRS',
    )
)

state0_teme = convert(
    Time(prior_time, format='datetime64', scale='utc'), 
    state0, 
    in_frame='ITRS', 
    out_frame='TEME',
)
orb = pyorb.Orbit(M0 = pyorb.M_earth, direct_update=True, auto_update=True, degrees=True, type='mean')
orb.cartesian = state0_teme.reshape(6,1)
print(orb)

mean0 = orb.kepler[:,0]
#The order is different (and remember its mean anomaly), but we still use SI units
tmp = mean0[4]
mean0[4] = mean0[3]
mean0[3] = tmp

spacetrack = json.load(open('spacetrack.json','r'))
tles = [(x['TLE_LINE1'],x['TLE_LINE2']) for x in spacetrack]
pop = tle_catalog(tles, cartesian=False) #for SGP4
pop.filter('oid', lambda x: x == 39771)
obj = pop.get_object(0)
print('Prior:')
print(obj.state[0])
print(obj.state[1])
mean00, B, epoch = obj.propagator.get_mean_elements(obj.state[0], obj.state[1])

mean0[:3] = mean00[:3] #take inclination and shape from prior

print(mean0)
print(epoch.iso)

# params = dict(SGP4_mean_elements=True, A=1.0, m=1.0, C_D=2.3, C_R=1.0)
params = dict(SGP4_mean_elements=True, B=B)

variables = ['a', 'e', 'i', 'raan', 'aop', 'mu']
dtype = [(name, 'float64') for name in variables]

state0_named = np.empty((1,), dtype=dtype)
step_arr = np.array([1e3,1e-1,1.,1.,1.,1.], dtype=np.float64)
step = np.empty((1,), dtype=dtype)

for ind, name in enumerate(variables):
    state0_named[name] = mean0[ind]
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
post.results.save('SGP4_mcmc_results.h5')

from pyod.posterior import _named_to_enumerated

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




print(post.results)

plot.orbits(post)
plot.residuals(post, [state0_named, post.results.MAP], ['Start', 'MAP'], ['-b', '-g'], absolute=False)
plot.residuals(post, [state0_named, post.results.MAP], ['Start', 'MAP'], ['-b', '-g'], absolute=True)

plt.show()


