import pathlib

import numpy as np
import scipy.io as sio

from pyod.sources import OpticalTracklet
from pyod.coordinates import geodetic2ecef

import sorts



#Knutsstorp
knt_geo = dict(
    lat=67+51/60+20.7/3600,
    lon=20+25/60+12.4/3600,
    alt=0.418e3,
)
knt_ecef = geodetic2ecef(radians=False, **knt_geo)

#Silkkimuotka
sil_geo = dict(
    lat=68+1/60+47/3600,
    lon=21+41/60+13.4/3600,
    alt=0.385e3,
)
sil_ecef = geodetic2ecef(radians=False, **sil_geo)

#Tjautjas
tja_geo = dict(
    lat=67+19/60+57.8/3600,
    lon=20+45/60+2.9/3600,
    alt=0.474e3,
)
tja_ecef = geodetic2ecef(radians=False, **tja_geo)

#Abisko
abi_geo = dict(
    lat=68+21/60+20.0/3600,
    lon=18+49/60+10.5/3600,
    alt=0.360e3,
)
abi_ecef = geodetic2ecef(radians=False, **abi_geo)

stations = {
    'K': {
        'name': 'Knutsstorp',
        'geo': knt_geo,
        'ecef': knt_ecef,
        'st': sorts.Station(**knt_geo, min_elevation=0.0, beam=None),
    },
    'S': {
        'name': 'Silkkimuotka',
        'geo': sil_geo,
        'ecef': sil_ecef,
        'st': sorts.Station(**sil_geo, min_elevation=0.0, beam=None),
    },
    'T': {
        'name': 'Tjautjas',
        'geo': tja_geo,
        'ecef': tja_ecef,
        'st': sorts.Station(**tja_geo, min_elevation=0.0, beam=None),
    },
    'A': {
        'name': 'Abisko',
        'geo': abi_geo,
        'ecef': abi_ecef,
        'st': sorts.Station(**abi_geo, min_elevation=0.0, beam=None),
    },
}


class ALID4DTrack(OpticalTracklet):

    ext = 'mat'

    def load(self):
        path = pathlib.Path(self.path.data)
        path = path.name

        mat = sio.loadmat(str(self.path.data))

        #old convention?
        # st_name = self.kwargs['station'] + 'l'

        #new convention?
        st_name = self.kwargs['station']

        #All functions use deg as default, data is in deg now
        data = np.empty((len(np.squeeze(mat['TimeS'])), ), dtype=OpticalTracklet.dtype)
        data['az'] = np.squeeze(mat['az' + st_name])
        data['el'] = 90 - np.squeeze(mat['ze' + st_name])

        data['az_sd'] = np.ones(data['az'].shape)*self.kwargs['az_sd']
        data['el_sd'] = np.ones(data['el'].shape)*self.kwargs['el_sd']

        date_lst = [path[10:14], path[14:16], path[16:18]]
        dt_start = np.datetime64('-'.join(date_lst))

        mat_T_us = np.squeeze(mat['TimeS'])*3600.0*1e6

        data['date'] = dt_start + mat_T_us.astype('timedelta64[us]')

        self.index = int(path[10:25].replace('T',''))
        self.meta = {
            'ecef': self.kwargs['ecef'], 
            'station': stations[self.kwargs['station']]['name'],
        }
        self.data = data





