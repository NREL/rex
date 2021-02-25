"""
Module to rechunk existing .h5 files
"""
from copy import deepcopy
import numpy as np
import sys


class ChunkSize:
    """
    Class to compute chunks of size 2MB
    """

    def __inti__(self, shape, dtype, chunk_size=2, data=None):
        """
        [summary]

        Parameters
        ----------
        shape : [type]
            [description]
        dtype : [type]
            [description]
        chunk_size : int, optional
            [description], by default 2
        data : [type], optional
            [description], by default None

        Returns
        -------
        [type]
            [description]
        """

    @staticmethod
    def get_data_size(data):
        """
        Get the size of the given data array in MB

        Parameters
        ----------
        data : ndarray
            Data array to chunk

        Returns
        -------
        float
            Data array size in MB
        """
        return sys.getsizeof(deepcopy(data)) * 10**-6

    @staticmethod
    def get_time_chunk(t_len, hourly_weeks=8):
        """
        Compute the size of the temporal chunks. Chunks are based on

        Parameters
        ----------
        t_len : int
            Length of temporal axis
        hourly_weeks : int, optional
            Number of weeks in a chunk of hourly data, by default 8

        Returns
        -------
        time_chunks : int
            Size of chunks along temporal axis
        """
        freq = t_len / 8760

        # Compute time chunks,
        time_chunk = np.ceil(hourly_weeks / freq) * 7 * 24 * freq

        return int(time_chunk)

    @staticmethod
    def calc_site_chunk_size(time_chunk, dtype, chunk_size=2):
        """
        Compute spatial chunk size given time chunk size

        Parameters
        ----------
        time_chunk : int
            Lenght of chunk along the temporal axis
        dtype : str | np.dtype
            Array data type (dtype)
        chunk_size : int, optional
            Chunk size in MB, by default 2

        Returns
        -------
        site_chunk : int
            Spatial chunk size or size of chunk along non-temporal axes
        """
        pixel_size = np.dtype(dtype).itemsize * 10**-6
        site_chunk = chunk_size // (time_chunk * pixel_size)

        return int(site_chunk)
