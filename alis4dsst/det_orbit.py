#!/usr/bin/env python

'''

'''

from pyod import PropagatorOrekit
from pyod import OptimizeLeastSquares, MCMCLeastSquares
from pyod import CameraStation

import pyod.plot as plot

import numpy as np
import matplotlib.pyplot as plt

from alis4d_source import sources, state0, prior_time, ecef_est


steps = int(1e5)


orekit_data = '/home/danielk/IRF/IRF_GITLAB/orekit_build/orekit-data-master.zip'

prop = PropagatorOrekit(
    orekit_data = orekit_data, 
    settings=dict(
        in_frame='ITRF',
        out_frame='ITRF',
        drag_force=False,
        radiation_pressure=False,
    )
)

params = dict(A= 0.1, m = 1.0)


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


post.results.save('mcmc_results.h5')

print(post.results)

plot.orbits(post)
plot.residuals(post, [state0_named, post.results.MAP], ['Start', 'MAP'], ['-b', '-g'], absolute=False)
plot.residuals(post, [state0_named, post.results.MAP], ['Start', 'MAP'], ['-b', '-g'], absolute=True)

plt.show()


