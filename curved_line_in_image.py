#!/usr/bin/env python
'''

Example command:

./curved_line_in_image.py detect -m radon -c -f 400 ~/data/alis4d/{2020-04-01T19.49.00K.fits,optpar_cam9_m3_f3_200401_200000.mat}

'''
from pathlib import Path
import argparse
import re

from astropy.io import fits
from astropy.time import Time, TimeDelta
from scipy.signal import convolve2d
from scipy.optimize import minimize
from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib import cm   
from matplotlib.animation import FuncAnimation
import numpy as np
from tqdm import tqdm

import sorts
import pyant
import pyorb

try:
    import skimage
    from skimage.transform import radon
    from skimage.transform import hough_line, hough_line_peaks
except ImportError:
    skimage = None

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

    if args.filter > 0:
        A[:, (A.shape[1]-args.filter):] = 0

    peaks = 2
    min_ind_d = 100

    A[A > 10] = 10
    A[:, 0] = 0
    A[:, -1] = 0
    A[0, :] = 0
    A[-1, :] = 0

    if args.method == 'radon':
        theta = np.linspace(0., 180., max(A.shape), endpoint=False)
        sinogram = radon(A, theta=theta)

        rad_map = (0, 180, -sinogram.shape[0]//2, sinogram.shape[0]//2)

        sgm = sinogram.reshape(sinogram.size)
        max_inds = np.argsort(sgm)[::-1]
        xi, yi = np.unravel_index(max_inds, sinogram.shape)
        dists = xi/sinogram.shape[1]*(rad_map[3] - rad_map[2]) + rad_map[2]
        angles = yi/sinogram.shape[0]*np.pi

        select = [0]
        while len(select) < peaks:
            ci = select[-1] + 1
            for xyi, x, y in zip(np.arange(ci, len(xi)), xi[ci:], yi[ci:]):
                d = np.sqrt((x - xi[np.array(select)])**2 + (y - yi[np.array(select)])**2)
                if np.all(d > min_ind_d):
                    select.append(xyi)
                    break
        select = np.array(select)
        dists = dists[select]
        angles = angles[select]
        xi = xi[select]
        yi = yi[select]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))

        ax1.set_title('Image')
        ax1.imshow(A, origin='lower', cmap='gray', vmax=10)

        for angle, dist in zip(angles, dists):
            xy0 = dist*np.array([
                np.cos(angle), 
                -np.sin(angle),  # -y, Radon transform specific
            ])
            (x0, y0) = xy0 + np.array(A.shape)//2
            ax1.axline((x0, y0), slope=np.tan(np.pi*0.5 - angle))
            ax1.plot(x0, y0, '.r')
            ax1.plot([A.shape[0]//2, x0], [A.shape[1]//2, y0], '--g')

        ax2.set_title('Radon transform (Sinogram)')
        ax2.set_xlabel('Line angle [deg]')
        ax2.set_ylabel('Line position [px]')
        ax2.imshow(sinogram, cmap='gray', aspect='auto', origin='lower', extent=rad_map)
        ax2.plot(np.degrees(angles), dists, '.r')

        fig.tight_layout()

        plt.show()

    elif args.method == 'hough':
        theta = np.linspace(-np.pi / 2, np.pi / 2, max(A.shape), endpoint=False)
        h, h_theta, d = hough_line(A, theta=theta)
        accum, angles, dists = hough_line_peaks(h, h_theta, d, num_peaks=peaks)

        hough_map = [-90, 90, d[-1], d[0]]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        ax1.imshow(A, cmap='gray', vmax=10)
        ax1.set_title('Image')

        for angle, dist in zip(angles, dists):
            (x0, y0) = dist*np.array([np.cos(angle), np.sin(angle)])
            ax1.axline((x0, y0), slope=np.tan(angle + np.pi/2))
            ax1.plot(x0, y0, '.r')
            ax1.plot([0, x0], [0, y0], '--g')
        ax1.set_xlim([0, A.shape[0]])
        ax1.set_ylim([0, A.shape[1]])

        ax2.imshow(h, cmap=cm.gray, aspect='auto', extent=hough_map)
        ax2.set_title('Hough transform')
        ax2.set_xlabel('Line angle [deg]')
        ax2.set_ylabel('Line position [px]')
        ax2.plot(np.degrees(angles), dists, '.r')

        plt.tight_layout()
        plt.show()

    elif args.method == 'orbit':

        shape = A.shape
        A = A.reshape(A.size)
        A_index = np.arange(A.size).reshape(*shape)

        u_edge = np.hstack([
            np.arange(shape[0], dtype=np.int64),
            np.full((shape[1],), shape[0] - 1, dtype=np.int64),
            np.arange(shape[0], dtype=np.int64)[::-1],
            np.full((shape[1],), 0, dtype=np.int64),
        ])
        v_edge = np.hstack([
            np.full((shape[0],), 0, dtype=np.int64),
            np.arange(shape[1], dtype=np.int64),
            np.full((shape[0],), shape[1] - 1, dtype=np.int64),
            np.arange(shape[1], dtype=np.int64)[::-1],
        ])

        az_edge, ze_edge = camera.inv_project_directions(
            u_edge, 
            v_edge, 
            imsize=shape,
        )

        local_sph_edge = np.vstack([
            az_edge, 
            np.pi/2 - ze_edge, 
            np.ones_like(az_edge),
        ])

        local_cart_edge = pyant.coordinates.sph_to_cart(local_sph_edge, radians=True)
        ecef_edge = sorts.frames.enu_to_ecef(
            knt_st.lat, 
            knt_st.lon, 
            knt_st.alt, 
            local_cart_edge, 
            radians=False,
        )

        range_i = [ecef_edge.shape[1] - 175]
        range_j = list(range(shape[0] + 300, shape[0] + 500, 1))

        metrics = np.zeros((len(range_i), len(range_j)))
        orbs = np.zeros((len(range_i), len(range_j), 4))

        pbar = tqdm(total=len(range_i)*len(range_j))
        for i_i, i_e in enumerate(range_i):
            for j_i, j_e in enumerate(range_j):

                ecef_edge1 = ecef_edge[:, i_e]
                ecef_edge1 = np.hstack([ecef_edge1, np.ones_like(ecef_edge1)])
                ecef_edge1.shape = (6, 1)
                ecef_edge2 = ecef_edge[:, j_e]
                ecef_edge2 = np.hstack([ecef_edge2, np.ones_like(ecef_edge2)])
                ecef_edge2.shape = (6, 1)

                nu_samples = 300

                orb = pyorb.Orbit(
                    M0=pyorb.M_earth, 
                    m=0, 
                    num=nu_samples, 
                    epoch=epoch, 
                    radians=True,
                    auto_update = False,
                    direct_update = False,
                )
                orb.a = 7.5e6
                orb.e = 0
                orb.i = 0
                orb.omega = 0
                orb.Omega = 0
                orb.anom = np.linspace(0.0, np.pi*2, nu_samples)
                ecef_st = np.reshape(knt_st.ecef, (3, 1))

                teme_st = sorts.frames.convert(
                    epoch, 
                    np.vstack([ecef_st, np.ones_like(ecef_st)]),
                    in_frame='ITRS', 
                    out_frame='TEME',
                )[:3, 0]

                teme_edge1 = sorts.frames.convert(
                    epoch, 
                    ecef_edge1,
                    in_frame='ITRS', 
                    out_frame='TEME',
                )[:3, 0]
                teme_edge2 = sorts.frames.convert(
                    epoch, 
                    ecef_edge2,
                    in_frame='ITRS', 
                    out_frame='TEME',
                )[:3, 0]

                def get_anom_inds(x):
                    orb0 = orb.copy()
                    orb0.i = x[0]
                    orb0.Omega = x[1]
                    orb0.calculate_cartesian()
                    st_teme_orb = orb0.cartesian[:3, :] - teme_st[:, None]
                    min1 = np.argmin(pyant.coordinates.vector_angle(teme_edge1, st_teme_orb))
                    min2 = np.argmin(pyant.coordinates.vector_angle(teme_edge2, st_teme_orb))
                    return min1, min2

                def optim_func(x):
                    orb0 = orb.copy()
                    orb0.i = x[0]
                    orb0.Omega = x[1]
                    orb0.calculate_cartesian()
                    st_teme_orb = orb0.cartesian[:3, :] - teme_st[:, None]
                    min1 = np.min(pyant.coordinates.vector_angle(teme_edge1, st_teme_orb))
                    min2 = np.min(pyant.coordinates.vector_angle(teme_edge2, st_teme_orb))
                    ang = min1 + min2
                    return ang

                min_res = minimize(
                    optim_func, np.array([np.pi/2, np.pi/4]), 
                    method='Nelder-Mead',
                )
                # print(min_res)

                min1, min2 = get_anom_inds(min_res.x)

                orbs[i_i, j_i, 0] = min1
                orbs[i_i, j_i, 1] = min2
                orbs[i_i, j_i, 2] = min_res.x[0]
                orbs[i_i, j_i, 3] = min_res.x[1]

                orb.i = min_res.x[0]
                orb.Omega = min_res.x[1]
                orb.calculate_cartesian()

                ecef_orb = sorts.frames.convert(
                    epoch, 
                    orb.cartesian,
                    in_frame='TEME', 
                    out_frame='ITRS',
                )[:3, :]

                orb.anom = np.linspace(orb.anom[min1], orb.anom[min2], nu_samples)
                orb.calculate_cartesian()
                ecef_orb = sorts.frames.convert(
                    epoch, 
                    orb.cartesian,
                    in_frame='TEME', 
                    out_frame='ITRS',
                )[:3, :]

                local_cart = knt_st.enu(ecef_orb)
                local_sph = pyant.coordinates.cart_to_sph(local_cart, radians=True)
                local_sph[1, :] = np.pi/2 - local_sph[1, :]

                u, v = camera.project_directions(
                    local_sph[0, :], 
                    local_sph[1, :],
                    imsize=shape,
                )
                keep = np.logical_and(u > 2, u < shape[0] - 2)
                keep = np.logical_and(
                    keep,
                    np.logical_and(v > 2, v < shape[1] - 2),
                )
                u = np.round(u[keep]).astype(np.int64)
                v = np.round(v[keep]).astype(np.int64)

                im_inds = np.unique(np.stack([
                    A_index[u, v],
                    A_index[u + 1, v],
                    A_index[u, v + 1],
                    A_index[u - 1, v],
                    A_index[u, v - 1],
                ]))

                metrics[i_i, j_i] = np.sum(A[im_inds])

                A_tmp = A.copy()
                A_tmp[im_inds] = 20
                A_tmp = A_tmp.reshape(*shape)

                pbar.update(1)

        pbar.close()

        max_metric = np.argmax(metrics)

        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.matshow(
            A.reshape(shape).T, 
            cmap='gray', norm=None, 
            vmax=10, origin='lower',
        )
        ax2.plot(np.array(range_j), metrics.flatten())

        plt.show()


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

    det_parser.add_argument(
        '-m', '--method', 
        choices=['radon', 'hough', 'orbit'],
        default='radon',
        help='Method to detect trace',
    )
    det_parser.add_argument(
        '-f', '--filter', 
        type=int,
        default=0,
        help='Remove part of the image from this row',
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
