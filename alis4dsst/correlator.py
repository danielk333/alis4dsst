import numpy as np
from astropy.time import Time, TimeDelta

import sorts
from sorts.population import tle_catalog
from sorts.frames import ITRS_to_geodetic, ecef_to_enu, cart_to_sph

from . import alis4d

def residual_distribution_metric(t, az, el, az_ref, el_ref):
    azx = np.cos(np.radians(az))
    azy = np.sin(np.radians(az))

    azx_ref = np.cos(np.radians(az_ref))
    azy_ref = np.sin(np.radians(az_ref))

    ang = np.abs(np.degrees(np.arccos(azx*azx_ref + azy*azy_ref)))

    residual_az_mu = np.mean(ang)

    # residual_az_mu = np.mean(az_ref - az)
    residual_el_mu = np.mean(np.abs(el_ref - el))
    metric = residual_az_mu + residual_el_mu
    return metric

def generate_measurements(state_ecef, rx_ecef, tx_ecef):
    lat, lon, alt = ITRS_to_geodetic(*rx_ecef.tolist(), radians=False).tolist()
    state_enu = ecef_to_enu(lat, lon, alt, state_ecef[:3,:] - rx_ecef[:,None], radians=False)

    sph = cart_to_sph(state_enu, radians=False)
    return sph[0,:], sph[1,:]


def correlate(sources, tles, propagator, **kwargs):

    logger = sorts.profiling.get_logger('correlate')

    if propagator.lower() == 'orekit':
        from sorts.propagator import Orekit

        prop_opts = dict(
            orekit_data = kwargs['orekit_data'], 
            settings=dict(
                in_frame='TEME',
                out_frame='ITRS',
                drag_force = kwargs.get('drag_force',False),
                radiation_pressure = False,
            ),
            logger = logger,
        )
        pop = tle_catalog(
            tles,
            propagator = Orekit,
            propagator_options = prop_opts,
            sgp4_propagation = False,
        )
    elif propagator.lower() == 'sgp4':
        prop_opts = dict(
            logger = logger,
        )
        pop = tle_catalog(
            tles,
            propagator_options = prop_opts,
            sgp4_propagation = True,
        )
    else:
        raise ValueError(f'Propagator "{propagator}" not recognized.')

    pop.out_frame = 'ITRS'

    if 'oids' in kwargs:
        pop.filter('oid', lambda oid: oid in kwargs['oids'])
        
    measurements = []
    for sc in sources:

        t = Time(sc.data['date'], format='datetime64', scale='utc')

        epoch = t.min()
        t = (t - epoch).sec

        dat = {
            'az': sc.data['az'],
            'el': sc.data['el'],
            't': t,
            'epoch': epoch,
            'tx': alis4d.stations[sc.kwargs['station']]['st'],
            'rx': alis4d.stations[sc.kwargs['station']]['st'],
        }
        measurements.append(dat)


    indecies, metric, cdat = sorts.correlate(
        measurements = measurements, 
        population = pop, 
        metric=residual_distribution_metric, 
        metric_reduce=lambda x,y: x+y,
        forward_model=generate_measurements,
        variables=['az','el'],
        n_closest=kwargs.get('n_closest', 1), 
        profiler=None, 
        logger=logger, 
        MPI=kwargs.get('MPI', False), 
    )

    return measurements, indecies, metric, cdat, pop