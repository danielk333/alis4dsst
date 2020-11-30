import pathlib
import pickle
import json
import sys

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from pyod import PosteriorParameters
from sorts.population import tle_catalog
import pyod.plot as odplot

import alis4dsst as a4



def cache_wrapper(func, cpath, override=False):
    def cached_func(*args, **kw):
        if cpath.is_file() and not override:
            with open(cpath, 'rb') as h:
                ret = pickle.load(h)
        else:
            ret = func(*args, **kw)
            with open(cpath, 'wb') as h:
                pickle.dump(ret, h)
        return ret
    return cached_func


def run_correlator(sources, tles, **kw):

    measurements, indecies, metric, cdat, pop = a4.correlate(
        sources, 
        tles, 
        propagator='SGP4',
        n_closest = len(tles),
        **kw
    )

    return measurements, indecies, metric, cdat, pop


def run_od(sources, obj, propagator, mcmc, time0=None, **kw):

    state0, params0, epoch0 = a4.od.get_mean_elements_start(tle_space_object = obj, start_epoch=time0)

    if propagator == 'sgp4':
        od_ = 'mean-elements'
    else:
        od_ = 'orekit'
        obj.out_frame = 'TEME'
        
        if time0 is None:
            state0 = obj.get_state(0.0)[:,0]
            epoch0 = obj.epoch
        else:
            state0 = obj.get_state(time0 - obj.epoch)[:,0]
            epoch0 = time0

    #todo: fix mcmc
    results = a4.od.determine_orbit(sources, start=state0, propagator=od_, epoch=epoch0, mcmc=mcmc, params=params0, **kw)
    state0_named= results['state0_named']
    post = results['post']
    variables = results['variables']

    return results



if __name__=='__main__':

    if len(sys.argv) < 2:
        fname = 'data/Sat_coord_20200401T194900a.mat'
    else:
        fname = sys.argv[1]

    #we know this one
    if fname == 'data/Sat_coord_20200401T195000b.mat':
        corr_kw = dict(oids = [39771])
    elif fname == 'data/Sat_coord_20200401T194900a.mat':
        #https://space.skyrocket.de/doc_sdat/hawkeye-pathfinder.htm
        #Did they do a maneuver???
        corr_kw = dict(oids = [43799])
    elif fname == 'data/Sat_coord_20200401T195200.mat':
        #lets find this automatically
        corr_kw = {}
        #corr_kw = dict(oids = [16182])
    else:
        corr_kw = {}

    if len(sys.argv) < 3:
        run_segments = ['corr', 'od', 'plot']
    else:
        run_segments = sys.argv[2:]
        run_segments = [x.lower().strip() for x in run_segments]

    if 'mcmc' in run_segments:
        mcmc = True
    else:
        mcmc = False

    if 'override' in run_segments:
        override = True
    else:
        override = False

    if 'sgp4' in run_segments:
        prop = 'sgp4'
        od_kw = {}
    else:
        prop = 'orekit'
        od_kw = dict(orekit_data = '/home/danielk/IRF/IRF_GITLAB/orekit_build/orekit-data-master.zip')

    err_fname = 'data/amb_function.mat'
    json_file = 'data/spacetrack.json'

    spacetrack = json.load(open(json_file,'r'))
    tles = [(x['TLE_LINE1'],x['TLE_LINE2']) for x in spacetrack]
    
    corr_cache = pathlib.Path('.'.join(fname.split('.')[:-1]) + '_correlation.pickle')
    od_cache = pathlib.Path('.'.join(fname.split('.')[:-1]) + f'_{prop}_od.pickle')
    wrapped_run_correlator = cache_wrapper(run_correlator, corr_cache, override=override)
    wrapped_run_od = cache_wrapper(run_od, od_cache, override=override)


    az_sd, el_sd, az_samps, el_samps = a4.io.load_sds(err_fname)
    sources, time0, state0 = a4.io.load_track(fname, az_sd, el_sd)

    if 'od' in run_segments or 'corr' in run_segments:
        measurements, indecies, metric, cdat, pop = wrapped_run_correlator(sources, tles, **corr_kw)

        for mi, meas in enumerate(measurements):
            print(f'Measurement {mi} time: {meas["epoch"]}')

        print(pop.print(n=indecies[0], fields=['line1']) + '\n')
        print(pop.print(n=indecies[0], fields=['line2']) + '\n')
        print(pop.print(n=indecies[0], fields=['A','m','d','C_D','C_R','BSTAR']) + '\n')

        if 'corr' in run_segments and 'plot' in run_segments:

            print('Metric, best')
            print(metric[0])
            print('Metric, runner ups')
            print(metric[1:100])
            
            fig, axes = plt.subplots(4,len(sources), sharex=True)
            
            for i in range(len(sources)):
                axes[0,i].set_title(a4.alis4d.stations[sources[i].kwargs['station']]['name'])
                a4.plots.correlation_azel(measurements[i], cdat[0][i], axes = [axes[j,i] for j in range(2)])
                a4.plots.correlation_resid(measurements[i], cdat[0][i], axes = [None]*2 + [axes[j,i] for j in range(2,4)])
            axes[0,0].legend()
            
            #plot Kiruna one
            a4.plots.correlation_track(measurements[2], cdat[0][2])

    if 'od' in run_segments:

        obj = pop.get_object(indecies[0])
        print('OD with prior:')
        print(obj)

        post, variables, state0_named = run_od(sources, obj, prop, mcmc, time0=None, **od_kw)

        if 'plot' in run_segments:

            print(post.results)

            for name in variables:
                print(f'{name}: {state0_named[name]} vs {post.results.MAP[name]}')

            a4.plots.model_to_data(state0_named, post)

            odplot.orbits(post)
            odplot.residuals(post, [state0_named, post.results.MAP], ['Start', 'MAP'], ['-b', '-g'], absolute=False)
            odplot.residuals(post, [state0_named, post.results.MAP], ['Start', 'MAP'], ['-b', '-g'], absolute=True)


    if 'plot' in run_segments:
        plt.show()