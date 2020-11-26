from pyod import PosteriorParameters
from pyod import propagate_results
from pyod import PropagatorOrekit
import pyod.plot as plot

import matplotlib.pyplot as plt

from alis4d_source import prior_time

results = PosteriorParameters.load_h5('mcmc_results.h5')

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

trace, inds = propagate_results(
    t = 60,
    date0 = prior_time, 
    results = results, 
    propagator = prop, 
    num = 1000, 
    params = dict(
        A= 0.1, 
        m = 1.0,
    ),
)


fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')
plot.earth_grid(ax)
ax.plot(results.trace[inds]['x'], results.trace[inds]['y'], results.trace[inds]['z'], '.b')
ax.plot(trace['x'], trace['y'], trace['z'], '.r')


# class Tmp_res:
#     def __init__(self,trace, variables):
#         self.trace = trace
#         self.variables = variables
# tmpr = Tmp_res(trace, results.variables)

# plot.scatter_trace(tmpr)


plt.show()