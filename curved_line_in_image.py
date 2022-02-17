#!/usr/bin/env python
from pathlib import Path
import argparse
import re

from astropy.io import fits
from astropy.time import Time, TimeDelta
from scipy.signal import convolve2d
from scipy.optimize import minimize
from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from tqdm import tqdm

import sorts
import pyant
import pyorb

try:
    import aida
    import aida.camera
except ImportError:
    aida = None

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


def read_optpar(file):
    fname = str(file.resolve())
    mat_data = loadmat(fname)
    optmod_pattern = r'_m[0-9]_'
    optmod_match = re.search(optmod_pattern, file.name)
    optmod = int(optmod_match[0][2:-1]) if optmod_match else None
    return mat_data['optpar'].flatten(), optmod
    

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
    

def get_Camera(optpar, optmod):
    camera = aida.camera.Camera(
        horizontalFocalLength=optpar[0],
        verticalFocalLength=optpar[1],
        rotationAngleAlpha=optpar[2],
        rotationAngleBeta=optpar[3],
        rotationAngleGamma=optpar[4],
        horizontalDisplacement=optpar[5],
        verticalDisplacement=optpar[6],
        pinholeDeviation=optpar[7],
    )
    if optmod == 3:
        camera.optical_transfer = aida.camera.model.alis3d
        camera.inverse_optical_transfer = aida.camera.invmodel.inv_alis3d
    return camera


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


def angle_to_pointing(orb, st, pointing, epoch):
    pos = sorts.frames.convert(
        epoch, 
        orb.cartesian, 
        in_frame='TEME', 
        out_frame='ITRS',
    )
    pos = pos[:3, :]

    local = st.enu(pos)
    ang = pyant.coordinates.vector_angle(pointing, local)
    return ang


def curve(t, orb, epoch, camera):

    # THIS WILL TAKE ORBIT AND CALCULATE PIXELS IN IMAGE WITH SIZE (using pixels + psf)
    ecef = prop.propagate(t, orb, epoch=epoch)

    local_cart = knt_st.enu(ecef[:3, :])
    local_sph = pyant.coordinates.cart_to_sph(local_cart, radians=True)
    local_sph[1, :] = np.pi/2 - local_sph[1, :]

    u, v = camera.project_directions(local_sph[0, :], local_sph[1, :])

    # TODO: ADD PSF HERE

    return np.stack([u, v], axis=1)


def curve_to_index(points, data_index, shape):
    inds = np.round(points).astype(np.int64)
    ok_inds = np.logical_and(inds[0, :] < shape[0], inds[1, :] < shape[1])
    ok_inds = np.logical_and(ok_inds, inds[0, :] > 0)
    ok_inds = np.logical_and(ok_inds, inds[1, :] > 0)
    inds = inds[:, ok_inds]
    inds = data_index[inds[0, :], inds[1, :]]
    inds = np.unique(inds)
    return inds


def curve_transform(img, index, t, orb, epoch, camera, shape):
    xy = curve(t, orb, epoch, camera)
    xy[0, :] *= shape[0]
    xy[1, :] *= shape[1]
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

    file = Path(args.optpar)
    optpar, optmod = read_optpar(file)

    assert optmod == 3, f'huh, optmod={optmod}'

    t_exp_start = np.arange(fits_h['NAXIS3'])*fits_h['CCDTKIN']
    t_exp_end = t_exp_start + fits_h['EXPTIME']
    exp_samples = 20

    t = np.hstack([
        np.linspace(t_exp_start[ind], t_exp_end[ind], exp_samples)
        for ind in range(fits_h['NAXIS3'])
    ])
    epoch = Time(fits_h['DATE-OBS'], format='isot', scale='utc')

    if fits_h['SITENAME'].lower().strip() == 'kirunaknutstorp':
        st = knt_st
    else:
        assert 0, f'warning unauthorized teleportation \
            detected to {fits_h["SITENAME"]}, error error'

    sph_pointing = np.array([
        float(fits_h['AZIMUTH']),
        90.0 - float(fits_h['ZENITANG']),
        1
    ])
    pointing = pyant.coordinates.sph_to_cart(sph_pointing, radians=True)

    camera = get_Camera(optpar, optmod)

    shape = A.shape
    A = A.reshape(A.size)
    A_index = np.arange(A.size).reshape(*shape)

    # orb = pyorb.Orbit(
    #     M0=pyorb.M_earth, m=0, 
    #     num=1, epoch=epoch, 
    #     radians=False,
    # )
    # orb.a = 7e6
    # orb.omega = 0

    # a_samp = np.linspace(6.7e6, 10e6, num=20)
    inc_samp = np.arange(60, 150, 1, dtype=np.float64)
    Omega_samp = np.arange(0, 360, 2.5, dtype=np.float64)
    nu_samp = np.arange(0, 360, 2.5, dtype=np.float64)
    inc, Omega, nu = np.meshgrid(
        inc_samp,
        Omega_samp,
        nu_samp,
        indexing='ij',
    )
    samples = inc.size
    samples_shape = inc.shape

    print(samples, samples_shape)

    inc = np.reshape(inc, (samples, ))
    Omega = np.reshape(Omega, (samples, ))
    nu = np.reshape(nu, (samples, ))

    orb = pyorb.Orbit(
        M0=pyorb.M_earth, 
        m=0, 
        num=samples, 
        epoch=epoch, 
        radians=False,
        auto_update = False,
        direct_update = False,
    )
    orb.a = 7e6
    orb.e = 0
    orb.i = inc
    orb.omega = 0
    orb.Omega = Omega
    orb.anom = nu
    orb.calculate_cartesian()

    # ang = angle_to_pointing(orb, st, pointing, epoch)

    # ang = np.reshape(ang, samples_shape)
    # inc = np.reshape(inc, samples_shape)
    # Omega = np.reshape(Omega, samples_shape)
    # nu = np.reshape(nu, samples_shape)

    # fig, ax = plt.subplots()
    # pm = ax.pcolormesh(inc[:, :, 0], Omega[:, :, 0], ang[:, :, 0])
    # cbar = fig.colorbar(pm, ax=ax)
    # pm.set_clim(0, 180)

    # def update(frame):
    #     ax.set_title(f'nu={nu_samp[frame]}')
    #     pm.set_array(ang[:, :, frame].flatten())
    #     return pm,

    # ani = FuncAnimation(
    #     fig, update, 
    #     frames=range(samples_shape[2]), 
    #     blit=False,
    # )
    # plt.show()

    # fig, ax = plt.subplots()
    # pm = ax.pcolormesh(inc[:, :, 0], Omega[:, :, 0], ang[:, :, 0])
    # fig.colorbar(pm, ax=ax)
    # plt.show()
    # exit()

    ecef = sorts.frames.convert(
        epoch, 
        orb.cartesian, 
        in_frame='TEME', 
        out_frame='ITRS',
    )
    local_cart = knt_st.enu(ecef[:3, :])
    local_sph = pyant.coordinates.cart_to_sph(local_cart, radians=True)
    local_sph[1, :] = np.pi/2 - local_sph[1, :]

    u, v = camera.project_directions(local_sph[0, :], local_sph[1, :])
    keep = np.logical_and(u > 0, u < 1)
    keep = np.logical_and(
        keep,
        np.logical_and(v > 0, v < 1),
    )
    u = u[keep]
    v = v[keep]
    xy = np.stack([u*shape[0], v*shape[1]], axis=1)

    print(xy.shape[1])

    fig, ax = plt.subplots()
    ax.plot(xy[0, :], xy[1, :], '.r')
    ax.matshow(
        data_dict['A'], 
        cmap='gray', norm=None, 
        vmax=10, origin='lower',
    )
    plt.show()

    # # orb0.i = orb.i + p[0]
    # # orb0.anom = orb.anom + p[1]
    # val = curve_transform(A, A_index, t, orb, epoch, camera, shape)
    # print(f'Curve transform val = {val}')

    # xy = curve(t, orb, epoch, camera)
    # xy[0, :] *= shape[0]
    # xy[1, :] *= shape[1]

    # fig, ax = plt.subplots()
    # ax.plot(xy[0, :], xy[1, :], '-r')
    # ax.matshow(
    #     data_dict['A'], 
    #     cmap='gray', norm=None, 
    #     vmax=10, origin='lower',
    # )
    # plt.show()


def print_fits(args):
    get_unit_regex = r'\[.*\]'
    header = fits.getheader(Path(args.path))
    for key in header:
        comment = header.comments[key]
        unit_match = re.search(get_unit_regex, comment)
        if unit_match:
            unit = unit_match[0][1:-1]
        else:
            unit = ''
        print(f'{key} = {header[key]} [{unit}] \\\\{comment}')


def main(input_args=None):
    parser = argparse.ArgumentParser(
        description='Analyse ALIS4D images for satellite tracks')
    subparsers = parser.add_subparsers(help='Action to perform', dest='command')
    plot_parser = subparsers.add_parser('plot', help='Plot the input image')
    det_parser = subparsers.add_parser('detect', help='Detect curve in image')
    fits_parser = subparsers.add_parser('fits', help='Print FITS header')
    opt_parser = subparsers.add_parser(
        'optpar', 
        help='Get Optical parameters from file',
    )

    for _pars in [plot_parser, det_parser]:
        _pars.add_argument(
            '-c', '--cache', 
            action='store_true',
            help='Cache results data along the way',
        )
        _pars.add_argument(
            '-C', '--clobber', 
            action='store_true',
            help='Override the cache results',
        )
        _pars.add_argument(
            '--seed', 
            type=int,
            default=837624764,
            help='Numpy seed',
        )
    for _pars in [fits_parser, plot_parser, det_parser]:
        _pars.add_argument(
            'path', 
            type=str,
            help='Path to the fits image to plot',
        )

    for _pars in [det_parser, opt_parser]:
        _pars.add_argument(
            'optpar', 
            type=str,
            help='Path to the optical parameter file',
        )

    if input_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(input_args)

    if args.command == 'plot':
        plot_data(args)
    elif args.command == 'detect':
        detect_trace(args)
    elif args.command == 'fits':
        print_fits(args)
    elif args.command == 'optpar':
        file = Path(args.optpar)
        optpar, optmod = read_optpar(file)
        print('Optpar = ')
        print(optpar)
        print(f'Optmod = {optmod}')


if __name__ == '__main__':
    main()
