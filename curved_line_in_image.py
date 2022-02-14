#!/usr/bin/env python
from pathlib import Path
import argparse

from astropy.io import fits
from astropy.time import Time, TimeDelta
from scipy.signal import convolve2d
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np

import sorts
import pyant
import pyorb

knt_geo = dict(
    lat=67+51/60+20.7/3600,
    lon=20+25/60+12.4/3600,
    alt=0.418e3,
)
knt_ecef = sorts.frames.geodetic_to_ITRS(**knt_geo)
knt_st = sorts.Station(**knt_geo, min_elevation=0.0, beam=None)

prop = sorts.propagator.Kepler(
    settings=dict(
        in_frame='TEME',
        out_frame='ITRS',
    ),
)


def find_sat_trace(in_image, set_value=10, mean_threshold=4):
    """
    find_sat_trace() -> im_def,im_def_filt, trace_im
    determens possible satellite trase from image sequence contened in 
    the in_image 3D nympy array
    
    Parametes:
         in_image - 3D nympy array with the image secuence

    Return:
         im_def - 3D numpy array with a difference of pixel intensities
                  for two sequential images from in_image
         im_def_filt - result of 2D filtering of the im_def array with
                       filter ones(3,3)
         trace_im - 3D numpy array with possible satellite trace
         
         Note: the first dimension of im_def, im_def_filt and trace_im is
               one less than in_image 

    Authors:
        Timothy Sergienko
        Daniel Kastinen
    
    """
    
    if len(in_image.shape) != 3:
        return print('Error: image array should be 3D')
    else:
        im_def = in_image[1:, :, :]-in_image[:-1, :, :]
        ff = np.ones((3, 3))/9
        trace_im = np.zeros_like(im_def)
        im_def_filt = np.zeros_like(im_def)
        
    for i in range(in_image.shape[0]-1):
        im_def_filt[i, :, :] = convolve2d(
            im_def[i, :, :], 
            ff, 
            mode='same', 
            boundary='symm',
            )   
        im1 = im_def_filt[i, :, :]
        i1 = (im1 > 0)
        mim = np.mean(im1[i1])
        i1 = ((im1 > mim*mean_threshold))        
        im1 = np.zeros_like(im_def[0, :, :])
        im1[i1] = set_value
        trace_im[i, :, :] = im1
    
    return im_def, im_def_filt, trace_im


def read_ALIS4D(file):
    """
    read_ALIS4D -> ALIS_im
    reads the sequence of images from the ALIS4D fits file
    
    Parameters:
         file - path of the ALIS4D fits file 
                          
    Return:
         ALIS_im - 3D nympy array with the image secuence.
    
    Authors:
        Timothy Sergienko
        Daniel Kastinen

    """
        
    fits_h = fits.getheader(file)
    
    try:
        black_level = fits_h['BZERO']
    except KeyError:
        pass

    ALIS_im = np.asarray(fits.getdata(file), dtype='float32')
    ALIS_im = ALIS_im + black_level
    
    return ALIS_im, fits_h
    

def generate_trace_data(path, cache=False, clobber=False, **kwargs):

    path = Path(path)
    cache_file = path.parent / (path.stem + '_trace.npz')
    if cache and cache_file.is_file() and not clobber:
        data_dict = np.load(cache_file)
        fits_h = fits.getheader(path)
    else:
        im, fits_h = read_ALIS4D(path)
        im_def, im_def_filt, trace_im = find_sat_trace(im, **kwargs)
        A = np.sum(trace_im, axis=0)
        data_dict = dict(
            A=A,
            im_def=im_def,
            im_def_filt=im_def_filt,
            trace_im=trace_im, 
        )

    if cache and (clobber or not cache_file.is_file()):
        np.savez(cache_file, **data_dict)

    return data_dict, fits_h


def locate_in_center(x, orb, st, pointing, epoch):
    orb0 = orb.copy()
    orb0.i = x[0]
    orb0.Omega = x[1]
    orb0.anom = x[2]
    pos = sorts.frames.convert(
        epoch, 
        orb0.cartesian, 
        in_frame='TEME', 
        out_frame='ITRS',
    )
    pos = pos.flatten()[:3]

    local = st.enu(pos)
    ang = pyant.coordinates.vector_angle(local, pointing)
    return ang


def curve(t, orb, epoch):

    # THIS WILL TAKE ORBIT AND CALCULATE PIXELS IN IMAGE WITH SIZE (using pixels + psf)
    ecef = prop.propagate(t, orb, epoch=epoch)

    local_cart = knt_st.enu(ecef[:3, :])
    local_sph = pyant.coordinates.cart_to_sph(local_cart)

    pixels = magic_function(local_sph)

    pixels =' widen stuff with psf'

    return pixels


def curve_to_index(points, data_index, shape):
    inds = np.round(points).astype(np.int64)
    ok_inds = np.logical_and(inds[0, :] < shape[0], inds[1, :] < shape[1])
    ok_inds = np.logical_and(ok_inds, inds[0, :] > 0)
    ok_inds = np.logical_and(ok_inds, inds[1, :] > 0)
    inds = inds[:, ok_inds]
    inds = data_index[inds[0, :], inds[1, :]]
    inds = np.unique(inds)
    return inds


def curve_transform(orb, stuff):

    # CALCULATE THE GENERALIZED CURVE TRANSFORM

    xy = curve(orb, data)
    xy_ind = curve_to_index(xy, index, shape)
    val = np.sum(img[xy_ind])
    return val


def plot_data(args):
    set_value = 10

    data_dict, fits_h = generate_trace_data(
        path=args.path, 
        cache=args.cache, 
        clobber=args.clobber,
        set_value=set_value, 
        mean_threshold=4,
    )
    plt.matshow(
        data_dict['A'], 
        cmap='gray', norm=None, 
        vmax=set_value, origin='lower',
    )
    plt.show()


def detect_trace(args):
    data_dict, fits_h = generate_trace_data(
        path=args.path, 
        cache=args.cache, 
        clobber=args.clobber,
        set_value=10, 
        mean_threshold=4,
    )
    A = data_dict['A'].copy()
    A[A > 10] = 10

    for key in fits_h:
        print(f'{key} = {fits_h[key]}')
    exit()

    framerate = float(fits_h['REXPSTR'].replace('Hz', ''))
    t_interval = fits_h['NAXIS3']/framerate
    epoch = Time(fits_h['DATE-OBS'], format='isot', scale='utc')
    point_az = float(fits_h['AZIMUTH'])
    point_el = 90.0 - float(fits_h['ZENITANG'])

    if fits_h['SITENAME'].lower().strip() == 'kirunaknutstorp':
        st = knt_st
    else:
        assert 0, 'wtf'

    shape = A.shape
    A = A.reshape(A.size)
    data_index = np.arange(A.size).reshape(*shape)

    num = int(np.sqrt(shape[0]**2 + shape[1]**2)) + 1

    orb = pyorb.Orbit(
        M0=pyorb.M_earth, m=0, 
        num=1, epoch=epoch, 
        radians=False,
    )
    orb.a = 7e6
    orb.e = 0
    orb.omega = 0

    it = 1
    x0 = np.array([51, 79, 158])
    min_res = minimize(
        locate_in_center, 
        x0, 
        args=(orb, st, pointing, epoch), 
        method='Nelder-Mead',
    )
    if args.v:
        print(f'{it}: [{min_res.fun}] {x0}')
    while it < 10 and min_res.fun > 0.1:
        it += 1
        x0 = np.random.rand(3)
        x0[0] *= 90
        x0[1:] *= 360
        min_res0 = minimize(
            locate_in_center, 
            x0, 
            args=(orb, st, pointing, epoch), 
            method='Nelder-Mead',
        )

        if args.v:
            print(f'{it}: [{min_res0.fun}] {x0} -> {min_res0.x}')
        if min_res0.fun < min_res.fun:
            min_res = min_res0
        if min_res.fun < 0.1:
            break
    if args.v:
        print(min_res)

    orb.i = min_res.x[0]
    orb.Omega = min_res.x[1]
    orb.anom = min_res.x[2]

    # Now this orb is at center of pointing at epoch

    # time extent of track from fits data
    t = stuff


    def opt_function(p):
        return -curve_transform(p, num, A, data_index, shape)

    np.random.seed(args.seed)

    best_result = None
    for ind in range(1000):
        x0 = [
            np.random.randint(shape[0]),
            np.random.randint(shape[1]),
            np.random.randint(shape[0]),
            np.random.randint(shape[1]),
            np.random.randint(shape[0]),
            np.random.randint(shape[1]),
        ]
        result = minimize(opt_function, x0, method='Nelder-Mead')
        if best_result is None:
            best_result = result
        
        if result.fun < best_result.fun:
            best_result = result

        print(f'iter={ind} -> best {best_result.fun} [current {result.fun}]')

    print(best_result)

    t = np.linspace(0, 1, num=num)
    xy = curve(t, *best_result.x)

    fig, ax = plt.subplots()
    ax.plot(xy[0, :], xy[1, :], '-r')
    ax.matshow(
        data_dict['A'], 
        cmap='gray', norm=None, 
        vmax=10, origin='lower',
    )
    plt.show()


def main(input_args=None):
    parser = argparse.ArgumentParser(
        description='Analyse ALIS4D images for satellite tracks')
    parser.add_argument(
        'path', 
        type=str,
        help='Path to the fits image to plot',
    )
    parser.add_argument(
        '-c', '--cache', 
        action='store_true',
        help='Cache results data along the way',
    )
    parser.add_argument(
        '-C', '--clobber', 
        action='store_true',
        help='Override the cache results',
    )
    subparsers = parser.add_subparsers(help='Action to perform', dest='command')
    plot_parser = subparsers.add_parser('plot', help='Plot the input image')
    det_parser = subparsers.add_parser('detect', help='Detect curve in image')
    det_parser.add_argument(
        '--seed', 
        type=int,
        default=837624764,
        help='Numpy seed',
    )

    if input_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(input_args)

    if args.command == 'plot':
        plot_data(args)
    elif args.command == 'detect':
        detect_trace(args)


if __name__ == '__main__':
    main()
