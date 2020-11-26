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
from sorts.frames import geodetic_to_ITRS, vector_angle

results = PosteriorParameters.load_h5('mcmc_results.h5')
date = Time(results.date)

MAP = np.array([results.MAP[0][var] for var in results.MAP.dtype.names])

print(MAP)
print(date.iso)
print(date.mjd)

kir_geo = dict(
    lat=67+50/60+26.6/3600,
    lon=20+24/60+40/3600,
    alt=0.425e3,
)
kir_ecef = geodetic_to_ITRS(radians=False, **kir_geo)

abi_geo = dict(
    lat=68+21/60+20.0/3600,
    lon=18+49/60+10.5/3600,
    alt=0.360e3,
)
abi_ecef = geodetic_to_ITRS(radians=False, **abi_geo)

sil_geo = dict(
    lat=68+1/60+47/3600,
    lon=21+41/60+13.4/3600,
    alt=0.385e3,
)
sil_ecef = geodetic_to_ITRS(radians=False, **sil_geo)

spacetrack = json.load(open('spacetrack.json','r'))

tles = [(x['TLE_LINE1'],x['TLE_LINE2']) for x in spacetrack]

pop = tle_catalog(tles) #for SGP4

#make sure its ecef
pop.out_frame = 'ITRS'

initial_zang = np.zeros((3, ), dtype=np.float64)
zenith_angle = np.zeros((len(tles), 3), dtype=np.float64)
states = np.zeros((6, len(tles)), dtype=np.float64)

pbar = tqdm(total=len(pop), ncols=100)
for ind, obj in enumerate(pop):
    states[:,ind] = obj.get_state(date).flatten()
    zenith_angle[ind,0] = vector_angle(kir_ecef,states[:3,ind])
    zenith_angle[ind,1] = vector_angle(abi_ecef,states[:3,ind])
    zenith_angle[ind,2] = vector_angle(sil_ecef,states[:3,ind])
    pbar.update(1)
pbar.close()

all_fov = np.argwhere(np.all(zenith_angle < 90, axis=1))
initial_zang[0] = vector_angle(kir_ecef,MAP[:3])
initial_zang[1] = vector_angle(abi_ecef,MAP[:3])
initial_zang[2] = vector_angle(sil_ecef,MAP[:3])

# for ind in all_fov.flatten():
#     for j in range(3):
#         print(f'{ind}: {zenith_angle[ind,j]} deg [{initial_zang[j] - zenith_angle[ind,j]}]')

best_matches = np.argsort(np.linalg.norm(initial_zang[None,:] - zenith_angle, axis=1))

for i in range(10):
    print(f'{i}: {zenith_angle[best_matches[i],:]} deg')
    print(pop.print(n=best_matches[i], fields = ['oid','mjd0'])+'\n')


# with h5py.File('zenith_angles.h5','w') as h:
#     h.create_dataset('zenith_angle', data=zenith_angle)
#     h.create_dataset('states', data=states)
