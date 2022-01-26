import numpy as np
import sorts
import pyant
from astropy.time import Time, TimeDelta
import astropy.coordinates as coord

# Knutsstorp
knt_geo = dict(
    lat=67+51/60+20.7/3600,
    lon=20+25/60+12.4/3600,
    alt=0.418e3,
)


# Uncomment this to try without sunlit consition
# class Camera(sorts.Station):
#     pass


class Camera(sorts.Station):
    def field_of_view(self, states, time):
        '''We can put custom FOV here to check sunlit condition
        '''

        sun = coord.get_sun(time)
        sun = sun.transform_to(coord.ITRS)
        ecef_sun_dir = np.stack([sun.x.value, sun.y.value, sun.z.value], axis=1).T
        approx_ecef_sun = ecef_sun_dir*149597871e3
        
        # Sides of triangle with corners in Earth center, sat and Sun
        a = np.linalg.norm(states[:3, :], axis=0)
        b = np.linalg.norm(approx_ecef_sun, axis=0)
        c = np.linalg.norm(states[:3, :] - approx_ecef_sun, axis=0)
        s = 0.5*(a + b + c)
        # Triangle area
        A = np.sqrt(s*(s - a)*(s - b)*(s - c))
        # Triangle height from base of edge between sat and sun up to earth center
        h = 2*A/c

        zenith = np.array([0, 0, 1], dtype=np.float64)

        enu = self.enu(states[:3, :])
        
        zenith_ang = pyant.coordinates.vector_angle(zenith, enu, radians=False)

        # If zenith angle is < 90 degrees, it is in the local horizon
        # if h < Earth radius, then LOS with sun is obscured
        in_fov = zenith_ang < 90.0
        in_sun = h > 6371e3
        check = np.logical_and(in_fov, in_sun)

        return check


st = Camera(**knt_geo, min_elevation=0.0, beam=None)

name = 'SENTINEL-1B'
state = np.array([
    '1 41456U 16025A   22026.13283885 -.00000102  00000-0 -11936-4 0  9993',
    '2 41456  98.1811  35.5401 0001308  77.7915 282.3439 14.59198667306439',
])
obj = sorts.SpaceObject(
    sorts.propagator.SGP4,
    propagator_options = dict(
        settings = dict(
            in_frame='TEME',
            out_frame='ITRF',
            tle_input=True,
        ),
    ),
    state=state,
    parameters = dict(
        d = 1.0,
    ),
)
t = np.arange(0, 24*3600.0, 10.0)
states = obj.get_state(t)

sat_dat = sorts.propagator.SGP4.get_TLE_parameters(*state.tolist())
epoch = Time(sat_dat['jdsatepoch'] + sat_dat['jdsatepochF'], format='jd', scale='utc')

passes = sorts.find_simultaneous_passes(
    t, 
    states, 
    [st], 
    cache_data=False, 
    fov_kw=dict(
        time=epoch + TimeDelta(t, format="sec"),
    ),
)

print(f'Passes for: {name}')

for ps in passes:
    str_ = f'Pass | Rise \
    {(epoch + TimeDelta(ps.start(), format="sec")).iso} \
    ({(ps.end() - ps.start())/60.0:.1f} min) \
    {(epoch + TimeDelta(ps.end(), format="sec")).iso} Fall'
    print(str_)
    
