import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import json
from matplotlib import cm
from astropy.time import Time
from tqdm import tqdm
import h5py

from pyod import PosteriorParameters
import sorts
from sorts.population import tle_catalog
from sorts.propagator import Orekit

orekit_data = '/home/danielk/IRF/IRF_GITLAB/orekit_build/orekit-data-master.zip'
logger = sorts.profiling.get_logger('orekit')
prop_opts = dict(
    orekit_data = orekit_data, 
    settings=dict(
        in_frame='TEME',
        out_frame='ITRS',
        drag_force = False,
        radiation_pressure = False,
    ),
    logger = logger,
)

results = PosteriorParameters.load_h5('mcmc_results.h5')
date = Time(results.date)

MAP = np.array([results.MAP[0][var] for var in results.MAP.dtype.names])

print(MAP)
print(date.iso)
print(date.mjd)

spacetrack = json.load(open('spacetrack.json','r'))

tles = [(x['TLE_LINE1'],x['TLE_LINE2']) for x in spacetrack]

# pop = tle_catalog(tles) #for SGP4
pop = tle_catalog(
    tles,
    propagator = Orekit,
    propagator_options = prop_opts,
    sgp4_propagation = False,
)

inds_ = np.where(pop.data['oid'] == 39771)
# pop.data = pop.data
print(inds_)
for ind in inds_:
    print(pop.print(n=ind, fields = ['oid','mjd0', 'm', 'A']))
    print(pop.data[ind]['mjd0'] - date.mjd)

exit()


# print(pop.print(n=slice(None,10), fields = ['oid','mjd0']))

match = np.zeros((len(tles),), dtype=np.float64)
states = np.zeros((6, len(tles)), dtype=np.float64)

pbar = tqdm(total=len(pop), ncols=100)
for ind, obj in enumerate(pop):
    try:
        states[:,ind] = obj.get_state(date).flatten()
        match[ind] = np.linalg.norm(MAP - states[:,ind])
    except:
        match[ind] = np.nan
    pbar.update(1)

match.sort()

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')
ax.plot(states[0,:], states[1,:], states[2,:],".b")
ax.plot([MAP[0]], [MAP[1]], [MAP[2]],"or")

plt.show()

with h5py.File('matches.h5','w') as h:
    h.create_dataset('match', data=match)
    h.create_dataset('states', data=states)
