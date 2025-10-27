import sys
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter as gf

# sys.path.append('../space/')
sys.path.append('../spok/')
from spok import utils as su


def listify(arg):
    if none_iterable(arg):
        return [arg]
    else:
        return arg


def none_iterable(*args):
    """
    return true if none of the arguments are either lists or tuples
    """
    return all([not isinstance(arg, list) and not isinstance(arg, tuple) and not isinstance(arg,
                                                                                            np.ndarray) and not isinstance(
        arg, pd.Series) for arg in args])


def index_isin(df1, df2):
    if isinstance(df1, list):
        return [df[df.index.isin(df2.index)] for df in df1]
    else:
        return df1[df1.index.isin(df2.index)]


def assert_regularity_grid(grid, treshold=1e-9):
    grad = np.gradient(grid)
    return np.sum(abs(np.diff(np.nan_to_num(grad)))) < treshold


def eliminateNoneValidValues(df):
    return df.replace([np.inf, -np.inf], np.nan).dropna()


def add_to_df(df, var, name):
    DF = df.copy()
    for n, v in zip(name, var):
        DF[n] = v
    return DF


def pandas_fill(arr):
    df = pd.DataFrame(arr)
    df.fillna(method='ffill', axis=1, inplace=True)
    out = df.values
    return out


def reshape_to_2Darrays(lst_arrays):
    lst_arrays = [np.array(su.listify(el)) for el in lst_arrays]
    if np.sum([lst_arrays[0].shape != el.shape for el in lst_arrays[1:]]):
        raise ValueError('All elements of lst_arrays must have the same shape')
    if len(lst_arrays[0].shape) == 1:
        a2d = np.array(lst_arrays).T
        old_shape = np.array(lst_arrays).shape
    else:
        a2d = np.asarray(lst_arrays)
        old_shape = a2d.shape
        a2d = a2d.T.ravel().reshape(np.prod(old_shape[1:]), old_shape[0])
    return a2d, old_shape


def reshape_to_original_shape(a2d, old_shape):
    if a2d.shape[0] == 1:
        lst_arrays = a2d.T
    else:
        lst_arrays = np.asarray(
            [a2d.reshape(old_shape[-1], np.prod(old_shape[:-1])).T[i::old_shape[0]] for i in range(old_shape[0])])
    return lst_arrays


def filter_nan_gaussian_conserving2(arr, sigma):
    """Apply a gaussian filter to an array with nans.

    Intensity is only shifted between not-nan pixels and is hence conserved.
    The intensity redistribution with respect to each single point
    is done by the weights of available pixels according
    to a gaussian distribution.
    All nans in arr, stay nans in gauss.
    """
    nan_msk = np.isnan(arr)

    loss = np.zeros(arr.shape)
    loss[nan_msk] = 1
    loss = gf(
        loss, sigma=sigma, mode='constant', cval=1)

    gauss = arr / (1 - loss)
    gauss[nan_msk] = 0
    gauss = gf(
        gauss, sigma=sigma, mode='constant', cval=0)
    gauss[nan_msk] = np.nan

    return gauss


def reshape_3d_to_2d(lst_arrays):
    lst_arrays = [np.array(su.listify(el)) for el in lst_arrays]
    if np.sum([lst_arrays[0].shape != el.shape for el in lst_arrays[1:]]):
        raise ValueError('All elements of lst_arrays must have the same shape')
    a2d = np.asarray([el.ravel() for el in lst_arrays]).T
    old_shape = np.array(lst_arrays).shape
    return a2d, old_shape


def reshape_2d_to_3d(a2d, old_shape):
    lst_arrays = np.asarray([a2d[:, i].reshape(old_shape[1:]) for i in range(old_shape[0])])
    return lst_arrays
