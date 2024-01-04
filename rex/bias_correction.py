# -*- coding: utf-8 -*-
"""
Module to perform bias correction of renewable energy resource data
"""
import numpy as np
import logging


logger = logging.getLogger(__name__)


def lin_irrad(ghi, dni, dhi, scalar=1, adder=0):
    """Correct GHI and DNI using linear correction factors. Both irradiance
    variables are corrected as ``irradiance * scalar + adder``. DHI is
    preserved based on the relationship ``dhi = ghi - (dni * cos(sza))``. Times
    when GHI and DNI are zero are preserved and negative values are protected
    against.

    Parameters
    ----------
    ghi : np.ndarray
        2D array of global horizontal irradiance values in shape (time, space)
    dni : np.ndarray
        2D array of direct normal irradiance values in shape (time, space)
    dhi : np.ndarray
        2D array of diffuse horizontal irradiance values in shape (time, space)
    scalar : np.ndarray
        1D array of linear scalar values in the shape (space,)
    adder : np.ndarray
        1D array of linear adder values in the shape (space,)

    Returns
    -------
    ghi : np.ndarray
        2D array of global horizontal irradiance values in shape (time, space)
    dni : np.ndarray
        2D array of direct normal irradiance values in shape (time, space)
    dhi : np.ndarray
        2D array of diffuse horizontal irradiance values in shape (time, space)
    """

    ghi_zeros = ghi == 0
    dni_zeros = dni == 0
    dhi_zeros = dhi == 0
    with np.errstate(divide='ignore', invalid='ignore'):
        cos_sza = (ghi - dhi) / dni

    ghi = ghi * scalar + adder
    dni = dni * scalar + adder

    ghi = np.maximum(0, ghi)
    dni = np.maximum(0, dni)

    ghi[ghi_zeros] = 0
    dni[dni_zeros] = 0

    with np.errstate(divide='ignore', invalid='ignore'):
        dhi = ghi - (dni * cos_sza)

    dhi[dni_zeros] = ghi[dni_zeros]
    dhi = np.maximum(0, dhi)
    dhi[dhi_zeros] = 0

    assert not np.isnan(dhi).any()

    return ghi, dni, dhi


def lin_ws(ws, scalar=1, adder=0):
    """Correct windspeed using linear correction factors. Windspeed is
    corrected as ``windspeed * scalar + adder`` with a minimum of zero.

    Parameters
    ----------
    ws : np.ndarray
        2D array of windspeed values in shape (time, space)
    scalar : np.ndarray
        1D array of linear scalar values in the shape (space,)
    adder : np.ndarray
        1D array of linear adder values in the shape (space,)

    Returns
    -------
    ws : np.ndarray
        2D array of windspeed values in shape (time, space)
    """
    ws = ws * scalar + adder
    ws = np.maximum(ws, 0)
    return ws
