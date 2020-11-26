from pyod.sources import OpticalTracklet
from pyod.coordinates import geodetic2ecef
from pyod import SourcePath
from pyod.plot import earth_grid

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio


kir_geo = dict(
    lat=67+50/60+26.6/3600,
    lon=20+24/60+40/3600,
    alt=0.425e3,
)
kir_ecef = geodetic2ecef(radians=False, **kir_geo)

abi_geo = dict(
    lat=68+21/60+20.0/3600,
    lon=18+49/60+10.5/3600,
    alt=0.360e3,
)
abi_ecef = geodetic2ecef(radians=False, **abi_geo)

sil_geo = dict(
    lat=68+1/60+47/3600,
    lon=21+41/60+13.4/3600,
    alt=0.385e3,
)
sil_ecef = geodetic2ecef(radians=False, **sil_geo)

geo_stations = [kir_geo, abi_geo, sil_geo]
ecef_stations = [kir_ecef, abi_ecef, sil_ecef]

#All functions use deg as default so convert to deg
class ALID4DTrack(OpticalTracklet):

    ext = 'mat'

    def load(self):
        path = self.path.data

        mat = sio.loadmat(str(path))

        data = np.empty((len(np.squeeze(mat['TimeS'])), ), dtype=OpticalTracklet.dtype)
        data['az'] = np.degrees(np.squeeze(mat['az' + self.kwargs['station'] + 'l']))
        data['el'] = 90 - np.degrees(np.squeeze(mat['ze' + self.kwargs['station'] + 'l']))

        data['az_sd'] = np.ones(data['az'].shape)*self.kwargs['az_sd']
        data['el_sd'] = np.ones(data['el'].shape)*self.kwargs['el_sd']

        date_lst = [path[10:14], path[14:16], path[16:18]]
        dt_start = np.datetime64('-'.join(date_lst))

        mat_T_us = np.squeeze(mat['TimeS'])*3600.0*1e6

        data['date'] = dt_start + mat_T_us.astype('timedelta64[us]')

        self.index = int(path[10:25].replace('T',''))
        self.meta = {'ecef': self.kwargs['ecef']}
        self.data = data



mat_err = sio.loadmat('amb_function.mat')
mat_err['amb_fun'] = mat_err['amb_fun'].flatten()/mat_err['amb_fun'].flatten().sum()
cumsum = np.cumsum(mat_err['amb_fun'])
mat_err['az_amb'] = np.degrees(mat_err['az_amb'].flatten())
mat_err['ze_amb'] = np.degrees(mat_err['ze_amb'].flatten())

n_est = int(1e6)
rng_ = np.random.rand(n_est)

rng_inds = np.zeros(rng_.shape, dtype=np.int64)

for i in range(n_est):
    rng_inds[i] = np.argmax(rng_[i] < cumsum)

az_sd = np.std(mat_err['az_amb'][rng_inds])
el_sd = np.std(mat_err['ze_amb'][rng_inds])

# print(az_sd)
# print(el_sd)

# fig, axes = plt.subplots(1, 2,figsize=(15,15), sharex=True)
# axes[0].hist(mat_err['az_amb'][rng_inds])
# axes[1].hist(mat_err['ze_amb'][rng_inds])
# plt.show()





kwargs = dict(
    path = SourcePath('Sat_coord_20200401T195015a.mat', 'file'),
    az_sd = az_sd,
    el_sd = el_sd,
)

sources = [
    ALID4DTrack(
        station = 'K',
        ecef = kir_ecef,
        **kwargs
    ),
    ALID4DTrack(
        station = 'A',
        ecef = abi_ecef,
        **kwargs
    ),
    ALID4DTrack(
        station = 'S',
        ecef = sil_ecef,
        **kwargs
    ),
]





mat = sio.loadmat('Sat_coord_20200401T195015a.mat')

ecef_est = np.zeros((3,mat['latS'].shape[0]))
for i in range(mat['latS'].shape[0]):
    ecef_tmp = geodetic2ecef(mat['latS'][i,0], mat['longS'][i,0], mat['altS'][i,0]*1e3, radians=False)
    ecef_est[:, i] = ecef_tmp



# fig = plt.figure(figsize=(15,15))
# ax = fig.add_subplot(111, projection='3d')
# earth_grid(ax)
# ax.plot(ecef_est[0,:], ecef_est[1,:], ecef_est[2,:])

# fig = plt.figure(figsize=(15,15))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(ecef_est[0,:], ecef_est[1,:], ecef_est[2,:])
# plt.show()


# from pyod import CameraStation

# fig, axes = plt.subplots(3, 2,figsize=(15,15), sharex=True)
# for i in range(3):
#     az_sim = np.zeros((ecef_est.shape[1],))
#     el_sim = np.zeros((ecef_est.shape[1],))
#     for j in range(ecef_est.shape[1]):
#         az_sim[j], el_sim[j] = CameraStation.generate_measurements(
#             ecef_est[:,j], 
#             ecef_stations[i], 
#             geo_stations[i]['lat'], 
#             geo_stations[i]['lon'],
#         )


#     daz = sources[i].data['az'] - az_sim
#     daz_tmp = np.mod(sources[i].data['az'] + 540.0, 360.0) - np.mod(az_sim + 540.0, 360.0)
#     inds_ = np.abs(daz) > np.abs(daz_tmp)
#     daz[inds_] = daz_tmp[inds_]

#     axes[i,0].plot(np.squeeze(mat['TimeS']), daz)
#     axes[i,1].plot(np.squeeze(mat['TimeS']), sources[i].data['el'] - el_sim)
# plt.show()



prior_ecef0 = geodetic2ecef(mat['latS'][0,0], mat['longS'][0,0], mat['altS'][0,0]*1e3, radians=False)
prior_ecef1 = geodetic2ecef(mat['latS'][1,0], mat['longS'][1,0], mat['altS'][1,0]*1e3, radians=False)

time_delta = (sources[0].data['date'][1] - sources[0].data['date'][0])/np.timedelta64(1, 's')
prior_vel = (prior_ecef1 - prior_ecef0)/time_delta

# print(time_delta)
# print(np.linalg.norm(prior_vel)*1e-3)
# print(prior_ecef0*1e-3)
# print(prior_ecef1*1e-3)

prior_time = sources[0].data['date'][0]

state0 = np.empty((6,))
state0[:3] = prior_ecef0
state0[3:] = prior_vel
