"""Main module."""
import sys
import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy.integrate import solve_ivp
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter as gf
from scipy.interpolate import LinearNDInterpolator
from structure_tensor import eig_special_2d, structure_tensor_2d
from datetime import timedelta, datetime
import collections
from scipy.interpolate import interp1d
from skimage.feature import peak_local_max
from sklearn.neighbors import KNeighborsRegressor
from shapely import Point, LineString

# sys.path.append('../space/')
# sys.path.append('../spok/')
sys.path.append('.')
sys.path.insert(0, './py_tsyganenko/build/')

from utilities import pandas_fill, reshape_to_2Darrays, reshape_to_original_shape, filter_nan_gaussian_conserving2, \
    assert_regularity_grid
import py_tsyganenko.Geopack as gp
import py_tsyganenko.Models as tm

from spok.models import planetary as smp
from spok import smath as sm
from spok.coordinates import coordinates as scc
from spok import utils as su


def date_for_recalc(date):
    '''
    Input :
        date : str
             format : 'DD-MM-YYYY hh:mm:ss'

    Output :
        year : int
        doy : int
        hour : int
        mins : int
        secs : int
    '''

    str_date = su.listify(date)
    date = pd.DatetimeIndex(str_date, dayfirst=True)
    year = pd.DatetimeIndex(date).year[0]
    doy = date.day_of_year[0]
    hour = date.hour[0]
    mins = date.minute[0]
    secs = date.second[0]
    return year, doy, hour, mins, secs


def tilt_geopack(date, vx=-400, vy=0, vz=0):
    year, doy, hour, mins, secs = date_for_recalc(date)
    gp.recalc(year, doy, hour, mins, secs, vx, vy, vz)
    return gp.GEOPACK1.PSI


def find_date_for_specific_tilt(wanted_tilt, freq_year='1D', freq_day='30S'):
    days = pd.date_range('01-01-2015', '01-01-2017', freq=freq_year)
    tilt_days = np.array([np.degrees(tilt_geopack(d, vx=-400)) for d in days])
    day = days[abs(tilt_days - wanted_tilt).argmin()]
    day_minutes = pd.date_range(day - timedelta(days=0.5), day + timedelta(days=0.5), freq=freq_day)
    tilt_minutes = np.array([np.degrees(tilt_geopack(d, vx=-400)) for d in day_minutes])
    minute = day_minutes[abs(tilt_minutes - wanted_tilt).argmin()]
    return str(minute), np.degrees(tilt_geopack(minute, vx=-400))


def gsm_to_gsw(date, xgsm, ygsm, zgsm, vx, vy, vz, vcoord='gsm'):
    '''
    if vx,vy, and vz are in gse coordinates, vcoord must be change to 'gse'
    '''
    if vy == 0 and vz == 0:
        xgsw, ygsw, zgsw = xgsm, ygsm, zgsm
    else:

        xgse, ygse, zgse = gsm_to_gse(date, xgsm, ygsm, zgsm)
        xgsw, ygsw, zgsw = gse_to_gsw(date, xgse, ygse, zgse, vx, vy, vz, vcoord=vcoord)

    return xgsw, ygsw, zgsw


def gsw_to_gsm(date, xgsw, ygsw, zgsw, vx, vy, vz, vcoord='gsm'):
    '''
    if vx,vy, and vz are in gse coordinates, vcoord must be change to 'gse'
    '''
    if vy == 0 and vz == 0:
        xgsm, ygsm, zgsm = xgsw, ygsw, zgsw
    else:
        xgse, ygse, zgse = gsw_to_gse(date, xgsw, ygsw, zgsw, vx, vy, vz, vcoord=vcoord)
        xgsm, ygsm, zgsm = gse_to_gsm(date, xgse, ygse, zgse)
    return xgsm, ygsm, zgsm


def gse_to_gsw(date, xgse, ygse, zgse, vx, vy, vz, vcoord='gsm'):
    '''
    if vx,vy, and vz are in gse coordinates, vcoord must be change to 'gse'
    '''

    vxgse, vygse, vzgse = velocity_to_gse(date, vx, vy, vz, vcoord=vcoord)
    pos_gse, old_shape = reshape_to_2Darrays([xgse, ygse, zgse])
    year, doy, hour, mins, secs = date_for_recalc(date)
    gp.recalc(year, doy, hour, mins, secs, vxgse, vygse, vzgse)
    pos_gsw = gp.gse_to_gsw(pos_gse)
    xgsw, ygsw, zgsw = reshape_to_original_shape(pos_gsw, old_shape)
    return xgsw, ygsw, zgsw


def gsw_to_gse(date, xgsw, ygsw, zgsw, vx, vy, vz, vcoord='gsm'):
    '''
    if vx,vy, and vz are in gse coordinates, vcoord must be change to 'gse'
    '''
    vxgse, vygse, vzgse = velocity_to_gse(date, vx, vy, vz, vcoord=vcoord)
    pos_gsw, old_shape = reshape_to_2Darrays([xgsw, ygsw, zgsw])
    year, doy, hour, mins, secs = date_for_recalc(date)
    gp.recalc(year, doy, hour, mins, secs, vxgse, vygse, vzgse)
    pos_gse = gp.gsw_to_gse(pos_gsw)
    xgse, ygse, zgse = reshape_to_original_shape(pos_gse, old_shape)
    return xgse, ygse, zgse


def gsm_to_gse(date, xgsm, ygsm, zgsm):
    pos_gsm, old_shape = reshape_to_2Darrays([xgsm, ygsm, zgsm])
    year, doy, hour, mins, secs = date_for_recalc(date)
    gp.recalc(year, doy, hour, mins, secs, -400, 0, 0)
    pos_gse = gp.gsw_to_gse(pos_gsm)
    xgse, ygse, zgse = reshape_to_original_shape(pos_gse, old_shape)
    return xgse, ygse, zgse


def gse_to_gsm(date, xgse, ygse, zgse):
    pos_gse, old_shape = reshape_to_2Darrays([xgse, ygse, zgse])
    year, doy, hour, mins, secs = date_for_recalc(date)
    gp.recalc(year, doy, hour, mins, secs, -400, 0, 0)
    pos_gse = gp.gse_to_gsw(pos_gse)
    xgsm, ygsm, zgsm = reshape_to_original_shape(pos_gse, old_shape)
    return xgsm, ygsm, zgsm


def velocity_to_gse(date, vx, vy, vz, vcoord='gsm'):
    '''
    will change the velocity from gsm to gse coordinates
    '''
    if vcoord == 'gsm':
        vxgse, vygse, vzgse = gsm_to_gse(date, vx, vy, vz)
        vxgse, vygse, vzgse = vxgse[0], vygse[0], vzgse[0]
    elif vcoord == 'gse':
        vxgse, vygse, vzgse = vx, vy, vz
    else:
        raise ValueError("Velocity coordinates must be in GSM (vcoord='gsm') or GSE (vcoord='gse')")
    return vxgse, vygse, vzgse


def coord_sys_to_gsw(date, x, y, z, vx, vy, vz, xcoord='gsm', vcoord='gsm'):
    if xcoord == 'gsm':
        xgsw, ygsw, zgsw = gsm_to_gsw(date, x, y, z, vx, vy, vz, vcoord=vcoord)
    elif xcoord == 'gse':
        xgsw, ygsw, zgsw = gse_to_gsw(date, x, y, z, vx, vy, vz, vcoord=vcoord)
    elif xcoord == 'gsw':
        xgsw, ygsw, zgsw = x, y, z
    else:
        raise ValueError(
            "Position coordinates must be in GSM (xcoord='gsm') or GSE (xcoord='gse') or GSW (xcoord='gsw')")
    return xgsw, ygsw, zgsw


def gsw_to_coord_sys(date, xgsw, ygsw, zgsw, vx, vy, vz, xcoord='gsm', vcoord='gsm'):
    if xcoord == 'gsm':
        x, y, z = gsw_to_gsm(date, xgsw, ygsw, zgsw, vx, vy, vz, vcoord=vcoord)
    elif xcoord == 'gse':
        x, y, z = gsw_to_gse(date, xgsw, ygsw, zgsw, vx, vy, vz, vcoord=vcoord)
    elif xcoord == 'gsw':
        x, y, z = xgsw, ygsw, zgsw
    else:
        raise ValueError(
            "Position coordinates must be in GSM (xcoord='gsm') or GSE (xcoord='gse') or GSW (xcoord='gsw')")
    return x, y, z


def coord_sys_to_gsm(date, x, y, z, vx, vy, vz, xcoord='gsm', vcoord='gsm'):
    if xcoord == 'gsw':
        xgsm, ygsm, zgsm = gsw_to_gsm(date, x, y, z, vx, vy, vz, vcoord=vcoord)
    elif xcoord == 'gse':
        xgsm, ygsm, zgsm = gse_to_gsm(date, x, y, z)
    elif xcoord == 'gsm':
        xgsm, ygsm, zgsm = x, y, z
    else:
        raise ValueError(
            "Position coordinates must be in GSM (xcoord='gsm') or GSE (xcoord='gse') or GSW (xcoord='gsw')")
    return xgsm, ygsm, zgsm


def gsm_to_coord_sys(date, xgsm, ygsm, zgsm, vx, vy, vz, xcoord='gsm', vcoord='gsm'):
    if xcoord == 'gsw':
        x, y, z = gsm_to_gsw(date, xgsm, ygsm, zgsm, vx, vy, vz, vcoord=vcoord)
    elif xcoord == 'gse':
        x, y, z = gsm_to_gse(date, xgsm, ygsm, zgsm)
    elif xcoord == 'gsm':
        x, y, z = xgsm, ygsm, zgsm
    else:
        raise ValueError(
            "Position coordinates must be in GSM (xcoord='gsm') or GSE (xcoord='gse') or GSW (xcoord='gsw')")
    return x, y, z


def mp_sibeck_for_t96(theta, phi, **kwargs):
    Pd = kwargs.get('Pd', 2.056)
    a0 = 0.14
    b0 = 18.2
    c0 = -217.2
    p0 = 2.04

    a = a0 * np.cos(theta) ** 2 + np.sin(theta) ** 2
    b = b0 * np.cos(theta)
    c = c0
    r = sm.resolve_poly2_real_roots(a, b, c)[0]
    r = r * (p0 / Pd) ** 0.158
    return scc.choice_coordinate_system(r, theta, phi, **kwargs)


def mp_sibeck1991_for_t96_tangents(theta, phi, **kwargs):
    theta = su.listify(theta)
    phi = su.listify(phi)
    Pd = kwargs.get("Pd", 2.056)

    a0 = 0.14
    b0 = 18.2
    c0 = -217.2
    p0 = 2.04
    # p0 = 2.0

    a = a0 * np.cos(theta) ** 2 + np.sin(theta) ** 2
    dadt = 2 * np.cos(theta) * np.sin(theta) * (1 - a0)

    b = b0 * np.cos(theta) * (p0 / Pd) ** (1 / 6)
    dbdt = -b0 * np.sin(theta) * (p0 / Pd) ** (1 / 6)

    c = c0 * (p0 / Pd) ** (1 / 3)
    dcdt = 0

    delta = b ** 2 - 4 * a * c
    ddeltadt = 2 * b * dbdt - 4 * dadt * c

    u = -b + np.sqrt(delta)
    dudt = -dbdt + ddeltadt / (2 * np.sqrt(delta))

    v = 2 * a
    dvdt = 2 * dadt

    r = sm.resolve_poly2_real_roots(a, b, c)[0]
    drdt = (dudt * v - dvdt * u) / v ** 2
    drdp = 0

    return smp.derivative_spherical_to_cartesian(r, theta, phi, drdt, drdp)


def mp_sibeck1991_for_t96_normal(theta, phi, **kwargs):
    tth, tph = mp_sibeck1991_for_t96_tangents(theta, phi, **kwargs)
    return smp.find_normal_from_tangents(tth, tph)


def find_crossing_normal_mp_sibeck_kf94(xmp, ymp, nx, ny, x0, phi):
    nx, ny = su.listify(nx), su.listify(ny)
    k = np.zeros_like(nx)
    k[ny != 0] = (nx[ny != 0] / ny[ny != 0])
    a = 1 / (2 * x0)
    b = k * np.sin(phi)
    c = xmp - x0 - k * ymp
    ryz = sm.resolve_poly2_real_roots(a, b, c)[0]

    x = x0 - ryz ** 2 / (2 * x0)
    y = ryz * np.sin(phi)
    z = ryz * np.cos(phi)
    return x, y, z


def magnetic_field_igrf(date, x, y, z, vx, vy, vz, xcoord='gsm', vcoord='gsm'):
    '''
    if x, y, and z are in gse or gsw coordinates, xcoord must be change to 'gse' or 'gsw'.
    if vx, vy, and vz are in gse coordinates, vcoord must be change to 'gse'.

    Output : bx,by,bz igrf in the same coordinate system than the positions (xcoord)
    '''
    vxgse, vygse, vzgse = velocity_to_gse(date, vx, vy, vz, vcoord=vcoord)
    xgsw, ygsw, zgsw = coord_sys_to_gsw(date, x, y, z, vxgse, vygse, vzgse, xcoord=xcoord, vcoord='gse')
    pos_gsw, old_shape = reshape_to_2Darrays([xgsw, ygsw, zgsw])
    year, doy, hour, mins, secs = date_for_recalc(date)
    gp.recalc(year, doy, hour, mins, secs, vxgse, vygse, vzgse)
    bgsw = gp.igrf_gsw(pos_gsw)
    bxgsw, bygsw, bzgsw = reshape_to_original_shape(bgsw, old_shape)
    bx, by, bz = gsw_to_coord_sys(date, bxgsw, bygsw, bzgsw, vxgse, vygse, vzgse, xcoord=xcoord, vcoord='gse')
    return bx, by, bz


def magnetic_field_t96(date, x, y, z, vx, vy, vz, pdyn, dst, byimf, bzimf, ps, xcoord='gsm', vcoord='gsm'):
    '''
    if x, y, and z are in gse or gsw coordinates, xcoord must be change to 'gse' or 'gsw'.
    if vx, vy, and vz are in gse coordinates, vcoord must be change to 'gse'.

    Output : bx,by,bz from tsyganenko's 1996 model in the same coordinate system than the positions (xcoord)
    '''
    vxgse, vygse, vzgse = velocity_to_gse(date, vx, vy, vz, vcoord=vcoord)
    xgsm, ygsm, zgsm = coord_sys_to_gsm(date, x, y, z, vxgse, vygse, vzgse, xcoord=xcoord, vcoord='gse')
    pos_gsm, old_shape = reshape_to_2Darrays([xgsm, ygsm, zgsm])
    year, doy, hour, mins, secs = date_for_recalc(date)
    gp.recalc(year, doy, hour, mins, secs, vxgse, vygse, vzgse)
    bgsm = tm.T96(pdyn, dst, byimf, bzimf, ps, pos_gsm)
    bxgsm, bygsm, bzgsm = reshape_to_original_shape(bgsm, old_shape)
    bx, by, bz = gsm_to_coord_sys(date, bxgsm, bygsm, bzgsm, vxgse, vygse, vzgse, xcoord=xcoord, vcoord='gse')
    return bx, by, bz


def kf94_field_mp_t96(theta, phi, pdyn, bximf, byimf, bzimf, bs_model=smp.bs_jelinek2012, coord=False):
    xmp, ymp, zmp = mp_sibeck_for_t96(theta, phi, Pd=pdyn)
    x0 = mp_sibeck_for_t96(0, 0, Pd=pdyn)[0]
    x1 = bs_model(0, 0)[0]
    nx, ny, nz = mp_sibeck1991_for_t96_normal(theta, phi)
    xmsh, ymsh, zmsh = find_crossing_normal_mp_sibeck_kf94(xmp, ymp, nx, ny, x0, phi)
    bxmsh, bymsh, bzmsh = smp.KF1994(xmsh, ymsh, zmsh, x0, x1, bximf, byimf, bzimf)
    if coord:
        return bxmsh, bymsh, bzmsh, xmsh, ymsh, zmsh
    else:
        return bxmsh, bymsh, bzmsh


def magnetospheric_field_t96(date, x, y, z, vx, vy, vz, pdyn, dst, byimf, bzimf, ps, xcoord='gsm', vcoord='gsm'):
    bxgsm0, bygsm0, bzgsm0 = magnetic_field_igrf(date, x, y, z, vx, vy, vz, xcoord=xcoord, vcoord=vcoord)
    bxgsm1, bygsm1, bzgsm1 = magnetic_field_t96(date, x, y, z, vx, vy, vz, pdyn, dst, byimf, bzimf, ps, xcoord=xcoord,
                                                vcoord=vcoord)
    bxmsp, bymsp, bzmsp = add_igrf_tmodel(bxgsm0, bygsm0, bzgsm0, bxgsm1, bygsm1, bzgsm1)
    return bxmsp, bymsp, bzmsp


def add_igrf_tmodel(bxgsm0, bygsm0, bzgsm0, bxgsm1, bygsm1, bzgsm1):
    return bxgsm0 + bxgsm1, bygsm0 + bygsm1, bzgsm0 + bzgsm1


def shear_angle(Bxmsp, Bymsp, Bzmsp, Bxmsh, Bymsh, Bzmsh):
    dp = Bxmsh * Bxmsp + Bymsh * Bymsp + Bzmsh * Bzmsp
    Bmsh = sm.norm(Bxmsh, Bymsh, Bzmsh)
    Bmsp = sm.norm(Bxmsp, Bymsp, Bzmsp)
    shear = np.degrees(np.arccos(dp / (Bmsh * Bmsp)))
    return shear


def shear_map_t96_kf94(theta, phi, date, pdyn, vsw, bximf, byimf, bzimf, dst, ps, bs_model=smp.bs_jelinek2012):
    '''
    Vy,Vz, Byimf, and Bzimf are set to zero to calculate the magnetospheric field.
    '''
    xmp, ymp, zmp = mp_sibeck_for_t96(theta, phi, Pd=pdyn)
    bxmsp, bymsp, bzmsp = magnetospheric_field_t96(date, xmp, ymp, zmp, -abs(vsw), 0, 0, pdyn, dst, 0, 0, ps,
                                                   xcoord='gsm', vcoord='gse')
    bxmsh, bymsh, bzmsh = kf94_field_mp_t96(theta, phi, pdyn, bximf, byimf, bzimf, bs_model=bs_model)
    shear = shear_angle(bxmsp, bymsp, bzmsp, bxmsh, bymsh, bzmsh)
    return xmp, ymp, zmp, shear


def remove_normal_to_shue98(xmp, ymp, zmp, vx, vy, vz):
    theta, phi = scc.cartesian_to_spherical(xmp, ymp, zmp)[1:]
    nx, ny, nz = smp.mp_shue1998_normal(theta, phi)
    bn = nx * vx + ny * vy + nz * vz
    return vx - bn * nx, vy - bn * ny, vz - bn * nz


def swi_to_negative_bximf(yy, zz, bx, by, bz):
    new_by = by.copy()
    new_bx, new_bz = interpolate_on_regular_grid(-yy, zz, [-bx, -bz], yy, zz)
    return new_bx, new_by, new_bz


def rotates_clock_angle(xmp, ymp, zmp, bx, by, bz, new_clock, old_clock):
    rotation_angle = new_clock - old_clock
    new_xmp, new_ymp, new_zmp = rotates_phi_angle(xmp, ymp, zmp, rotation_angle)
    bx_new, by_new, bz_new = rotates_phi_angle(bx, by, bz, rotation_angle)
    return new_xmp, new_ymp, new_zmp, bx_new, by_new, bz_new


def make_regular_grid(**kwargs):
    xlim = kwargs.get('xlim', (-20, 20))
    ylim = kwargs.get('ylim', (-20, 20))
    nb_pts = kwargs.get('nb_pts', 401)
    x = np.linspace(xlim[0], xlim[1], nb_pts)
    y = np.linspace(ylim[0], ylim[1], nb_pts)
    xx, yy = np.meshgrid(x, y, indexing=kwargs.get('indexing', 'xy'))
    return xx, yy


def make_regular_interpolation(x, y, qty, new_x, new_y):
    qty_2d = reshape_to_2Darrays([qty])[0]
    xy = reshape_to_2Darrays([x, y])[0]
    interp = KNeighborsRegressor(n_neighbors=25, weights='distance')
    interp.fit(xy, qty_2d)
    reg_qty = np.array([interp.predict(p) for p in np.array([new_x, new_y]).T]).T[0]
    del interp
    rmax = np.max(sm.norm(x, y, 0))
    r = sm.norm(new_x, new_y, 0)
    reg_qty[r > rmax] = np.nan
    return reg_qty


def interpolate_on_regular_grid(x, y, qties, xx, yy, **kwargs):
    if (not isinstance(qties, list)) and (not isinstance(qties, np.ndarray)):
        qties = [qties]
    return [make_regular_interpolation(x, y, q, xx, yy, **kwargs) for q in qties]


def make_gaussian_filter_for_non_regular_grid(x, y, qties, xx, yy, sigma, **kwargs):
    if (not isinstance(qties, list)) and (not isinstance(qties, np.ndarray)):
        qties = [qties]
    qties = interpolate_on_regular_grid(x, y, qties, xx, yy, **kwargs)
    g_qties = make_gaussian_filter(qties, sigma)
    return g_qties


def make_gaussian_filter(qties, sigma):
    return [filter_nan_gaussian_conserving2(q, sigma) for q in qties]


def rotates_phi_angle(x, y, z, angle):
    r, th, ph = scc.cartesian_to_spherical(x, y, z)
    return scc.spherical_to_cartesian(r, th, ph + angle)


def gradient_2d_grid(yy, zz, qty, indexing='xy'):
    if indexing == 'ij':
        yaxis = 0
        zaxis = 1
    else:
        yaxis = 1
        zaxis = 0

    grad = np.gradient(qty)
    dy = np.gradient(yy)[yaxis]
    dz = np.gradient(zz)[zaxis]
    if (np.sum(np.diff(dy)) > 1e-10) or (np.sum(np.diff(dz)) > 1e-10):
        print('The grid is not regular')
    return grad[yaxis] / dy, grad[zaxis] / dz


def find_potential_saddle_and_extremum_points(x, y, qty, n=3, threshold=1e-1, indexing='xy'):
    if n < 2:
        raise ValueError('n should be at superior or equal to 2')
    grad = gradient_2d_grid(x, y, qty, indexing=indexing)
    norm_grad = sm.norm(0, grad[0], grad[1])
    i, j = np.where(norm_grad < threshold)
    if len(i) == 0:
        raise ValueError(
            'No potential saddle points has been found. Should increase the threshold or verify the indexing')
    else:
        return i, j


def find_potential_saddle_points_with_hessian(i, j, x, y, qty, n=3, threshold=1e-1, rlim_sdl=8):
    i_sdls = []
    j_sdls = []
    qty_sdls = []
    for k in range(len(i)):
        matrice_nn = qty[i[k] - n:i[k] + n + 1, j[k] - n:j[k] + n + 1]
        if matrice_nn.shape != (2 * n + 1, 2 * n + 1):
            continue
        eig1, eig2 = hessian_matrix_eigvals(hessian_matrix(matrice_nn, sigma=np.std(matrice_nn) * 0.01, order='rc'))
        if (np.sum((eig1 * eig2) < 0) > 0.95 * matrice_nn.size) & (
            sm.norm(0, x[i[k], j[k]], y[i[k], j[k]]) <= rlim_sdl):
            i_sdls.append(i[k])
            j_sdls.append(j[k])
            qty_sdls.append(qty[i[k], j[k]])
    if len(qty_sdls) == 0:
        raise ValueError(
            'No saddle point has been found. Try to increase the threshold or rlim_sdl, or try to decrease n.')
    else:
        return i_sdls, j_sdls, qty_sdls


def make_hessian_e1_vector(x, y, qty):
    Hrr, Hrc, Hcc = hessian_matrix(qty)
    mat = np.zeros((len(x), len(y), 2, 2))
    mat[:, :, 0, 0] = Hrr
    mat[:, :, 1, 0] = Hrc
    mat[:, :, 0, 1] = Hrc
    mat[:, :, 1, 1] = Hcc
    hess_val, hess_vec = np.linalg.eigh(mat)
    return hess_vec[:, :, :, 0]


def make_hessian_e2_vector(x, y, qty):
    Hrr, Hrc, Hcc = hessian_matrix(qty)
    mat = np.zeros((len(x), len(y), 2, 2))
    mat[:, :, 0, 0] = Hrr
    mat[:, :, 1, 0] = Hrc
    mat[:, :, 0, 1] = Hrc
    mat[:, :, 1, 1] = Hcc
    hess_val, hess_vec = np.linalg.eigh(mat)
    return hess_vec[:, :, :, 1]


def make_linear_interpolator(x, y, qty):
    arr2d = reshape_to_2Darrays([x, y, qty])[0]
    return LinearNDInterpolator(arr2d[:, :2], arr2d[:, -1])


def make_hessian_e1_interpolator(x, y, qty, indexing='xy'):
    if indexing == 'xy':
        yaxis = 0
        zaxis = 1
    else:
        yaxis = 1
        zaxis = 0
    e1 = make_hessian_e1_vector(x, y, qty)
    e1x = make_linear_interpolator(x, y, e1[:, :, yaxis])
    e1y = make_linear_interpolator(x, y, e1[:, :, zaxis])
    return e1x, e1y


def make_hessian_e2_interpolator(x, y, qty, indexing='xy'):
    if indexing == 'xy':
        yaxis = 0
        zaxis = 1
    else:
        yaxis = 1
        zaxis = 0
    e2 = make_hessian_e2_vector(x, y, qty)
    e2x = make_linear_interpolator(x, y, e2[:, :, yaxis])
    e2y = make_linear_interpolator(x, y, e2[:, :, zaxis])
    return e2x, e2y


def make_gradient_interpolator(x, y, qty, indexing='xy'):
    grad = gradient_2d_grid(x, y, qty, indexing=indexing)
    gx = make_linear_interpolator(x, y, grad[0])
    gy = make_linear_interpolator(x, y, grad[1])
    return gx, gy


def structure_tensor_eig_vec(qty, sigma=0.5, rho=1, indexing='xy'):
    if indexing == 'ij':
        yaxis = 0
        zaxis = 1
    else:
        yaxis = 1
        zaxis = 0

    S = structure_tensor_2d(qty, sigma, rho)
    st_val, st_vec = eig_special_2d(S)
    return st_vec[yaxis], st_vec[zaxis]


def make_structure_tensor_vec_interpolator(x, y, qty, sigma=0.5, rho=1, indexing='xy'):
    vec = structure_tensor_eig_vec(qty, sigma=sigma, rho=rho, indexing=indexing)
    st_x = make_linear_interpolator(x, y, vec[0])
    st_y = make_linear_interpolator(x, y, vec[1])
    return st_x, st_y


def outofbounds_with_grad(t, pos, interxy, rlim):
    if np.sqrt(pos[0] ** 2 + pos[1] ** 2) > rlim:
        v = 0
    elif sm.norm(0, interxy[0](pos[0], pos[1]), interxy[1](pos[0], pos[1])) < 1e-5:
        v = 0
    else:
        v = 1
    return v


outofbounds_with_grad.terminal = True


def get_line_with_gradient(intergrad, interhess2,
                           x0=0,
                           y0=0,
                           t0=0,
                           tfinal=100,
                           fac=0.1,
                           max_step=0.05, first_step=0.05, rlim=15,
                           outofbounds=outofbounds_with_grad):
    def vel(t,  # pseudo time
            pos,  # x and y positions
            intergrad, rlim):  # eigenvector interpolators in x and y directions:   # some arbitrary magnification coef
        vv = [(intergrad[0](pos[0], pos[1])),
              (intergrad[1](pos[0], pos[1]))]
        return vv

    dx, dy = interhess2[0]([x0, y0])[0], interhess2[1]([x0, y0])[0]

    return solve_ivp(vel, [t0, tfinal], [x0 + (fac * dx), y0 + (fac * dy)],
                     args=(intergrad, rlim),
                     method="BDF", events=outofbounds_with_grad, max_step=max_step, first_step=max_step).y


def find_position_of_all_saddle_points(xx, yy, qty, n=5, threshold=1e-1, rlim_sdl=16):
    i, j = find_potential_saddle_and_extremum_points(xx, yy, qty, n=n, threshold=threshold)
    i_sdls, j_sdls, qty_sdls = find_potential_saddle_points_with_hessian(i, j, xx, yy, qty, n=n, threshold=threshold,
                                                                         rlim_sdl=rlim_sdl)
    xs = np.asarray([xx[i, j] for i, j in zip(i_sdls, j_sdls)])
    ys = np.asarray([yy[i, j] for i, j in zip(i_sdls, j_sdls)])
    return xs, ys, qty_sdls


def clustering_close_points(x, y, qty, distance=1):
    coords = list(np.asarray([x, y, qty]).T)
    C = []
    while len(coords):
        locus = coords.pop()
        i_cluster = [i for i, x in enumerate(coords) if sm.norm(0, locus[0] - x[0], locus[1] - x[1]) <= distance]
        cluster = [coords[i] for i in i_cluster]
        C.append(cluster + [locus])
        for i in i_cluster[::-1]:
            coords.pop(i)
    return C


def select_point_in_cluster(cluster_pts, selection='mean'):
    if selection == 'mean':
        x = np.array([np.mean(np.array(c)[:, 0]) for c in cluster_pts])
        y = np.array([np.mean(np.array(c)[:, 1]) for c in cluster_pts])
        qty = np.array([np.mean(np.array(c)[:, 2]) for c in cluster_pts])
    if selection == 'median':
        x = np.array([np.median(np.array(c)[:, 0]) for c in cluster_pts])
        y = np.array([np.median(np.array(c)[:, 1]) for c in cluster_pts])
        qty = np.array([np.median(np.array(c)[:, 2]) for c in cluster_pts])
    if selection == 'min_qty':
        x, y, qty = np.asarray([c[np.argmin(np.asarray(c)[:, 2])] for c in cluster_pts]).T
    if selection == 'max_qty':
        x, y, qty = np.asarray([c[np.argmax(np.asarray(c)[:, 2])] for c in cluster_pts]).T
    return x, y, qty


def remove_point_under_threshold(xf, yf, qf, thrsh_qty):
    return xf[qf > thrsh_qty], yf[qf > thrsh_qty], qf[qf > thrsh_qty]


def find_position_valid_saddle_points(xx, yy, qty, qty_thresh=20, n_hess=3, grad_thresh=0.1, rlim_sdl=15,
                                      distance_cluster=1):
    xs, ys, qty_sdls = find_position_of_all_saddle_points(xx, yy, qty, n=n_hess, threshold=grad_thresh,
                                                          rlim_sdl=rlim_sdl)
    cluster_sddl = clustering_close_points(xs, ys, qty_sdls, distance=distance_cluster)
    xs, ys, qty_sdls = select_point_in_cluster(cluster_sddl, selection='mean')
    xf, yf, qf = remove_point_under_threshold(xs, ys, qty_sdls, qty_thresh)
    return xf, yf, qf


def find_max_points(xx, yy, qty, rlim_max=9, min_distance=1):
    coord = peak_local_max(qty, min_distance=min_distance)
    xm = np.asarray([xx[c[0], c[1]] for c in coord])
    ym = np.asarray([yy[c[0], c[1]] for c in coord])
    qm = np.asarray([qty[c[0], c[1]] for c in coord])
    r = sm.norm(xm, ym, 0)
    xm, ym, qm = xm[r <= rlim_max], ym[r <= rlim_max], qm[r <= rlim_max]
    return xm, ym, qm


def find_critic_points(xx, yy, qty, qty_thresh=None, n_hess=5, grad_thresh=0.1, distance_cluster=1, distance_btw_max=3,
                       rlim_max=15, rlim_sdl=15):
    xm, ym, qm = find_max_points(xx, yy, qty, rlim_max=rlim_max, min_distance=distance_btw_max)
    if qty_thresh is None:
        qty_thresh = np.min(qm) / 2
    xs, ys, qs = find_position_valid_saddle_points(xx, yy, qty, qty_thresh=qty_thresh, n_hess=n_hess,
                                                   grad_thresh=grad_thresh, rlim_sdl=rlim_sdl,
                                                   distance_cluster=distance_cluster)
    return (xm, ym, qm), (xs, ys, qs)


def remove_point_under_threshold(xf, yf, qf, thrsh_qty):
    return xf[qf > thrsh_qty], yf[qf > thrsh_qty], qf[qf > thrsh_qty]


def outofbounds_anti_branch(t, pos, interxy, i_fac, rlim):
    if np.sqrt(pos[0] ** 2 + pos[1] ** 2) > rlim:
        v = 0
    else:
        v = 1
    return v


outofbounds_anti_branch.terminal = True


def integrate_antiparallel_branch_with_hess2(interhess2, i_fac,
                                             x0=0,
                                             y0=0,
                                             t0=0,
                                             tfinal=100,
                                             max_step=0.05, first_step=0.05, rlim=15,
                                             outofbounds=outofbounds_anti_branch):
    def vel(t,  # pseudo time
            pos,  # x and y positions
            interhess2, i_fac,
            rlim):  # eigenvector interpolators in x and y directions:   # some arbitrary magnification coef
        fac_x, fac_y = i_fac.predict([[pos[0], pos[1]]])[0]
        if fac_x == 0:
            fac_x = 1
        if fac_y == 0:
            fac_y = 1
        vv = [np.sign(fac_x) * abs(interhess2[0](pos[0], pos[1])),
              np.sign(fac_y) * abs(interhess2[1](pos[0], pos[1]))]

        return vv

    return solve_ivp(vel, [t0, tfinal], [x0, y0],
                     args=(interhess2, i_fac, rlim),
                     method="BDF", events=outofbounds_anti_branch, max_step=max_step, first_step=max_step).y


def outofbounds(t, pos, interxy, rlim):
    if np.sqrt(pos[0] ** 2 + pos[1] ** 2) > rlim:
        v = 0
    else:
        v = 1
    return v


outofbounds.terminal = True


def get_line_with_hess2(interhess2,
                        x0=0,
                        y0=0,
                        t0=0,
                        tfinal=100,
                        fac=1,
                        max_step=0.05, first_step=0.05, rlim=15,
                        outofbounds=outofbounds):
    def vel(t,  # pseudo time
            pos,  # x and y positions
            interhess2, rlim):  # eigenvector interpolators in x and y directions:   # some arbitrary magnification coef
        vv = [fac * (interhess2[0](pos[0], pos[1])),
              fac * (interhess2[1](pos[0], pos[1]))]

        return vv

    return solve_ivp(vel, [t0, tfinal], [x0, y0],
                     args=(interhess2, rlim),
                     method="BDF", events=outofbounds, max_step=max_step, first_step=max_step).y


def find_position_of_list_in_list_of_lists_of_various_length(lst, lsts):
    if len(np.asarray(lst).shape) < 2:
        lst = [lst]
    idx = np.where(
        [all([collections.Counter(ll[i]) == collections.Counter(lst[i]) for i in range(len(lst))]) for ll in lsts])[0]
    return idx


def find_lines_following_gradient_from_saddle_points(xs, ys, igrad, ie2, fac_e2=0.15, rlim=16):
    lines = []
    for x0, y0 in zip(xs, ys):
        part1 = get_line_with_gradient(igrad, ie2, x0=x0, y0=y0, fac=fac_e2, rlim=rlim)
        part2 = get_line_with_gradient(igrad, ie2, x0=x0, y0=y0, fac=-fac_e2, rlim=rlim)
        if sm.norm(0, part1[0][-1], part1[1][-1]) > sm.norm(0, part2[0][-1], part2[1][-1]):
            line = np.concatenate([part2[0][::-1], part1[0]]), np.concatenate([part2[1][::-1], part1[1]])
        elif sm.norm(0, part1[0][-1], part1[1][-1]) < sm.norm(0, part2[0][-1], part2[1][-1]):
            line = np.concatenate([part1[0][::-1], part2[0]]), np.concatenate([part1[1][::-1], part2[1]])
        lines.append(line)
    return lines


def concat_lines_in_one(lines, dl=0.5):
    line = lines.pop()
    cnt = 0
    while (len(lines) > 0) and (cnt < (3 * len(lines))):
        for l in (lines):
            if ((abs(line[0][0] - l[0][0]) < dl) & (abs(line[1][0] - l[1][0]) < dl)):
                line = np.concatenate([l[0][::-1], line[0]]), np.concatenate([l[1][::-1], line[1]])
                lines.pop(find_position_of_list_in_list_of_lists_of_various_length(l, lines)[0])
            elif ((abs(line[0][0] - l[0][-1]) < dl) & (abs(line[1][0] - l[1][-1]) < dl)):
                line = np.concatenate([l[0], line[0]]), np.concatenate([l[1], line[1]])
                lines.pop(find_position_of_list_in_list_of_lists_of_various_length(l, lines)[0])
            elif ((abs(line[0][-1] - l[0][0]) < dl) & (abs(line[1][-1] - l[1][0]) < dl)):
                line = np.concatenate([line[0], l[0]]), np.concatenate([line[1], l[1]])
                lines.pop(find_position_of_list_in_list_of_lists_of_various_length(l, lines)[0])
            elif ((abs(line[0][-1] - l[0][-1]) < dl) & (abs(line[1][-1] - l[1][-1]) < dl)):
                line = np.concatenate([line[0], l[0][::-1]]), np.concatenate([line[1], l[1][::-1]])
                lines.pop(find_position_of_list_in_list_of_lists_of_various_length(l, lines)[0])
            else:
                pass
        cnt += 1
    return line


def find_max_line_from_saddles(xs, ys, igrad, ie2, fac_e2=0.15, rlim=16):
    lines = find_lines_following_gradient_from_saddle_points(xs, ys, igrad, ie2, fac_e2=fac_e2, rlim=rlim)
    line = concat_lines_in_one(lines, dl=0.5)
    return line


def identify_antiparallel_branches_max_points(xm, ym):
    sign = np.sign(xm)
    branch1 = xm[sign < 0], ym[sign < 0]
    branch2 = xm[sign > 0], ym[sign > 0]
    return branch1, branch2


def organize_max_points_in_antipara_branch(x, y):
    xt, yt = list(x), list(y)
    y = [yt.pop(np.argmin(np.abs(xt)))]
    x = [xt.pop(np.argmin(np.abs(xt)))]
    while len(xt):
        i = np.argmin(sm.norm(0, x[-1] - np.array(xt), y[-1] - np.array(yt)))
        x.append(xt.pop(i))
        y.append(yt.pop(i))
    return np.array(x), np.array(y)


def find_orientation_integration_hess2_antipara(xm, ym):
    i_fac = KNeighborsRegressor(n_neighbors=1, weights='distance', n_jobs=1)
    i_fac.fit(np.asarray(su.make_center_bins([xm, ym], dd=1)).T, np.asarray([np.diff(xm), np.diff(ym)]).T)
    return i_fac


def backward_integrate_antiparallel_branch(xm, ym, ie2, i_fac, rlim=16):
    branch = integrate_antiparallel_branch_with_hess2(ie2, i_fac, tfinal=-100, x0=xm[-1], y0=ym[-1], rlim=rlim)
    branch = branch[0][:np.argmin(sm.norm(0, xm[0] - branch[0], ym[0] - branch[1])) + 1], branch[1][:np.argmin(
        sm.norm(0, xm[0] - branch[0], ym[0] - branch[1])) + 1]
    return branch


def foward_integrate_antiparallel_branch(xm, ym, ie2, i_fac, rlim=16):
    branch = integrate_antiparallel_branch_with_hess2(ie2, i_fac, x0=xm[0], y0=ym[0], rlim=rlim)
    branch = branch[0][:np.argmin(sm.norm(0, xm[-1] - branch[0], ym[-1] - branch[1])) + 1], branch[1][:np.argmin(
        sm.norm(0, xm[-1] - branch[0], ym[-1] - branch[1])) + 1]
    return branch


def get_potentiel_branches(xm, ym, ie2, i_fac, rlim=16):
    branch0 = foward_integrate_antiparallel_branch(xm, ym, ie2, i_fac, rlim=rlim)
    branch1 = backward_integrate_antiparallel_branch(xm, ym, ie2, i_fac, rlim=rlim)
    return branch0, branch1


def select_most_valid_branch(xm, ym, list_branches, dr=0.25):
    passing_by_max_points = [np.sum([any(sm.norm(0, x - b[0], y - b[1]) < dr) for x, y in zip(xm, ym)]) for b in
                             list_branches]
    return list_branches[np.argmax(passing_by_max_points)]


def get_antiparallel_branch(xm, ym, ie2, rlim=16, dr=0.25):
    xm, ym = organize_max_points_in_antipara_branch(xm, ym)
    i_fac = find_orientation_integration_hess2_antipara(xm, ym)
    xm, ym = xm[1:-1], ym[1:-1]
    potential_branch = get_potentiel_branches(xm, ym, ie2, i_fac, rlim=rlim)
    branch = select_most_valid_branch(xm, ym, potential_branch, dr=dr)
    return branch


def get_antiparallel_branches_shear(xm, ym, ie2, rlim=16):
    max_branches = identify_antiparallel_branches_max_points(xm, ym)
    branches = [get_antiparallel_branch(b[0], b[1], ie2, rlim=rlim) for b in max_branches if len(b[0]) > 0]
    return branches


def cut_antipara_branch_for_jonction_with_other_line(branch, line):
    xl, yl, xb, yb = line[0], line[1], branch[0], branch[1]
    i = np.argmin(abs(xb))
    if any(sm.norm(0, xl[0] - xb, yl[0] - yb) < 0.25):
        j = np.argmin(sm.norm(0, xl[0] - xb, yl[0] - yb))
    else:
        j = np.argmin(sm.norm(0, xl[-1] - xb, yl[-1] - yb))
    if i > 5:
        branch = xb[:j], yb[:j]
    else:
        branch = xb[j:], yb[j:]
    return branch


def concat_compo_antiparallel_lines(compo_line, branches):
    lines = [compo_line] + [cut_antipara_branch_for_jonction_with_other_line(b, compo_line) for b in branches]
    line = concat_lines_in_one(lines, dl=1)
    return line


def make_anti_parallel_lines(xx, yy, qty, rlim_sdl=9, rlim=16, fac_e2=0.1, n_hess=5, grad_thresh=0.1, indexing='xy'):
    ie2 = make_hessian_e2_interpolator(xx, yy, qty, indexing=indexing)
    (xm, ym, qm), (xs, ys, qs) = find_critic_points(xx, yy, qty, qty_thresh=0, n_hess=n_hess, grad_thresh=grad_thresh,
                                                    distance_cluster=1, distance_btw_max=1, rlim_max=rlim,
                                                    rlim_sdl=rlim_sdl)
    branches = get_antiparallel_branches_shear(xm, ym, ie2, rlim=rlim)
    return branches


def find_lines_following_hessian_from_max_point(xm, ym, ie2, rlim=16):
    part1 = get_line_with_hess2(ie2, x0=xm, y0=ym, fac=-1, rlim=rlim)
    part2 = get_line_with_hess2(ie2, x0=xm, y0=ym, fac=1, rlim=rlim)
    if sm.norm(0, part1[0][-1], part1[1][-1]) > sm.norm(0, part2[0][-1], part2[1][-1]):
        line = np.concatenate([part2[0][::-1], part1[0]]), np.concatenate([part2[1][::-1], part1[1]])
    elif sm.norm(0, part1[0][-1], part1[1][-1]) < sm.norm(0, part2[0][-1], part2[1][-1]):
        line = np.concatenate([part1[0][::-1], part2[0]]), np.concatenate([part1[1][::-1], part2[1]])
    return line


def make_max_qty_line_from_saddle_or_max_points(xx, yy, qty0, qty_thresh=None, rlim=16, fac_e2=0.1, n_hess=5,
                                                grad_thresh=0.1, norm_qty=True, indexing='xy'):
    if norm_qty:
        qty = qty0 * 10 / np.median(qty0)
    else:
        qty = qty0
    (xm, ym, qm), (xs, ys, qs) = find_critic_points(xx, yy, qty, qty_thresh=qty_thresh, n_hess=n_hess,
                                                    grad_thresh=grad_thresh, distance_cluster=1, distance_btw_max=1,
                                                    rlim_max=rlim, rlim_sdl=rlim)
    ie2 = make_hessian_e2_interpolator(xx, yy, qty, indexing=indexing)
    if len(xs) >= 1:
        igrad = make_gradient_interpolator(xx, yy, qty, indexing=indexing)
        line = find_max_line_from_saddles(xs, ys, igrad, ie2, fac_e2=fac_e2, rlim=rlim)
    else:
        if len(xm) > 1:
            print('Warning more than one max point without a saddle point : check the validity of the parameters')
        line = find_lines_following_hessian_from_max_point(xm[0], ym[0], ie2, rlim=rlim)
    return line


def make_max_line_from_saddle_points(xx, yy, qty, qty_thresh=None, rlim=16, fac_e2=0.1, n_hess=5, grad_thresh=0.1,
                                     indexing='xy'):
    (xm, ym, qm), (xs, ys, qs) = find_critic_points(xx, yy, qty, qty_thresh=qty_thresh, n_hess=n_hess,
                                                    grad_thresh=grad_thresh, distance_cluster=1, distance_btw_max=1,
                                                    rlim_max=rlim, rlim_sdl=rlim)
    ie2 = make_hessian_e2_interpolator(xx, yy, qty, indexing=indexing)
    igrad = make_gradient_interpolator(xx, yy, qty, indexing=indexing)
    line = find_max_line_from_saddles(xs, ys, igrad, ie2, fac_e2=fac_e2, rlim=rlim)
    return line


def make_max_qty_from_max_point(xx, yy, qty, rlim=16, rlim_subsolar=None, indexing='xy'):
    if rlim_subsolar:
        xm, ym, qm = find_max_points(xx, yy, qty, rlim_max=rlim_subsolar)
    else:
        xm, ym, qm = find_max_points(xx, yy, qty, rlim_max=rlim)
    xm, ym = xm[np.argmax(qm)], ym[np.argmax(qm)]
    ie2 = make_hessian_e2_interpolator(xx, yy, qty, indexing=indexing)
    line = find_lines_following_hessian_from_max_point(xm, ym, ie2, rlim=rlim)
    return line


def interpolate_max_points_antiparallel_branches(xm, ym, n_pts=100):
    xm, ym = organize_max_points_in_antipara_branch(xm, ym)
    dl = [LineString([[xm[0], ym[0]], [x, y]]).length for x, y in zip(xm, ym)]
    return np.interp(np.linspace(0, np.max(dl), n_pts), dl, xm), np.interp(np.linspace(0, np.max(dl), n_pts), dl, ym)


def eliminate_points_outside_rlim(x, y, rlim=16):
    r = sm.norm(x, y, 0)
    return x[r <= rlim], y[r <= rlim]


def get_antiparallel_branches_shear_from_interp_max_point(xm, ym, rlim=16):
    max_branches = identify_antiparallel_branches_max_points(xm, ym)
    branches = [interpolate_max_points_antiparallel_branches(b[0], b[1]) for b in max_branches if len(b[0]) > 0]
    branches = [eliminate_points_outside_rlim(b[0], b[1], rlim=rlim) for b in branches if len(b[0]) > 0]
    return branches


def make_max_shear_line(xx, yy, qty, rlim_sdl=9, rlim=16, fac_e2=0.1, n_hess=5, grad_thresh=0.1, indexing='xy'):
    ie2 = make_hessian_e2_interpolator(xx, yy, qty, indexing=indexing)
    igrad = make_gradient_interpolator(xx, yy, qty, indexing=indexing)
    (xm, ym, qm), (xs, ys, qs) = find_critic_points(xx, yy, qty, qty_thresh=0, n_hess=n_hess, grad_thresh=grad_thresh,
                                                    distance_cluster=1, distance_btw_max=3, rlim_max=rlim,
                                                    rlim_sdl=rlim_sdl)
    # branches = get_antiparallel_branches_shear(xm,ym,ie2,rlim=rlim) old version where the anti-parallel branches are obtained with the hessian eigenvector 2
    branches = get_antiparallel_branches_shear_from_interp_max_point(xm, ym, rlim=rlim)
    compo_line = find_max_line_from_saddles(xs, ys, igrad, ie2, fac_e2=fac_e2, rlim=rlim)
    line = concat_compo_antiparallel_lines(compo_line, branches)
    return line


def make_max_qty_line_from_saddle_or_max_points(xx, yy, qty, r_subsolar=7, rlim_max=20, qty_thresh=None, rlim=16,
                                                fac_e2=0.1, n_hess=5, grad_thresh=0.1, indexing='xy', verbose=True):
    if np.nanmax(qty) < 10:
        qty = qty * 10 / np.median(qty)

    xm, ym, qm = find_max_points(xx, yy, qty, rlim_max=rlim_max)
    xm, ym, qm = xm[sm.norm(0, xm, ym) <= rlim], ym[sm.norm(0, xm, ym) <= rlim], qm[sm.norm(0, xm, ym) <= rlim]
    if sm.norm(0, xm[np.argmax(qm)], ym[np.argmax(qm)]) < r_subsolar:
        if verbose:
            print('From max point')
        line = make_max_qty_from_max_point(xx, yy, qty, rlim=rlim, indexing='xy')
    else:
        if verbose:
            print('From saddle points')
        line = make_max_line_from_saddle_points(xx, yy, qty, qty_thresh=qty_thresh, rlim=rlim, fac_e2=fac_e2,
                                                n_hess=n_hess, grad_thresh=grad_thresh, indexing='xy')
    return line


def organize_points_in_line(x, y, cond=0):
    xt, yt = list(x), list(y)
    x = [xt.pop(cond)]
    y = [yt.pop(cond)]
    while len(xt):
        i = np.argmin(sm.norm(0, x[-1] - np.array(xt), y[-1] - np.array(yt)))
        x.append(xt.pop(i))
        y.append(yt.pop(i))
    return np.array(x), np.array(y)


def organize_points_in_line_from_y_min(x, y):
    xt, yt = list(x), list(y)
    x = [xt.pop(np.argmin(yt))]
    y = [yt.pop(np.argmin(yt))]
    while len(xt):

        i = np.argmin(sm.norm(0, x[-1] - np.array(xt), y[-1] - np.array(yt)))
        if yt[i] < y[-1]:
            xt.pop(i)
            yt.pop(i)
        else:
            x.append(xt.pop(i))
            y.append(yt.pop(i))
    return np.array(x), np.array(y)


def concatenate_problematic_lines_from_y_min(lines):
    xl, yl = [], []
    for t in lines:
        xl = np.concatenate([xl, t[0]])
        yl = np.concatenate([yl, t[1]])
    xl, yl = organize_points_in_line_from_y_min(xl, yl)
    return xl, yl


def make_max_line_from_problematic_saddle_points(xx, yy, qty, qty_thresh=None, rlim=16, fac_e2=0.1, n_hess=5,
                                                 grad_thresh=0.1, pop_sddl=None, indexing='xy'):
    if np.nanmax(qty) < 10:
        qty = qty * 10 / np.median(qty)
    (xm, ym, qm), (xs, ys, qs) = find_critic_points(xx, yy, qty, qty_thresh=qty_thresh, n_hess=n_hess,
                                                    grad_thresh=grad_thresh, distance_cluster=1, distance_btw_max=1,
                                                    rlim_max=rlim, rlim_sdl=rlim)
    if pop_sddl is not None:
        print(xs, ys, qs)
        xs, ys, qs = list(xs), list(ys), list(qs)
        pop_sddl.sort()
        for i in pop_sddl[::-1]:
            xs.pop(i)
            ys.pop(i)
            qs.pop(i)
        print(xs, ys, qs)
    ie2 = make_hessian_e2_interpolator(xx, yy, qty, indexing=indexing)
    igrad = make_gradient_interpolator(xx, yy, qty, indexing=indexing)
    lines = find_lines_following_gradient_from_saddle_points(xs, ys, igrad, ie2, fac_e2=fac_e2, rlim=rlim)
    line = concatenate_problematic_lines_from_y_min(lines)
    return line


def make_max_qty_from_saddle_pts_with_hessian(xx, yy, qty, qty_thresh=None, rlim=16, n_hess=5, grad_thresh=0.1,
                                              pop_sddl=None, indexing='xy'):
    if np.nanmax(qty) < 10:
        qty = qty * 10 / np.median(qty)
    (xm, ym, qm), (xs, ys, qs) = find_critic_points(xx, yy, qty, qty_thresh=qty_thresh, n_hess=n_hess,
                                                    grad_thresh=grad_thresh, distance_cluster=1, distance_btw_max=1,
                                                    rlim_max=rlim, rlim_sdl=rlim)
    if pop_sddl is not None:
        print(xs, ys, qs)
        xs, ys, qs = list(xs), list(ys), list(qs)
        pop_sddl.sort()
        for i in pop_sddl[::-1]:
            xs.pop(i)
            ys.pop(i)
            qs.pop(i)
        print(xs, ys, qs)
    ie2 = make_hessian_e2_interpolator(xx, yy, qty, indexing=indexing)
    lines = [find_lines_following_hessian_from_max_point(x, y, ie2, rlim=rlim) for x, y in zip(xs, ys)]
    line = concatenate_problematic_lines_from_y_min(lines)
    return line
