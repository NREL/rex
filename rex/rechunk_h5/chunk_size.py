"""
Module to rechunk existing .h5 files
"""
from abc import ABC
from math import ceil
import numpy as np


class BaseChunkSize(ABC):
    """
    Base class to compute chunk size
    """
    def __init__(self):
        """
        Parameters
        ----------
        chunk_size : int, optional
            Chunk size in MB, by default 2
        """
        self._chunks = None

    @property
    def chunks(self):
        """
        Dataset chunk size along all axis

        Returns
        -------
        tuple
        """
        return self._chunks


class TimeseriesChunkSize(BaseChunkSize):
    """
    Compute Timeseries chunks based on dtype, and weeks_per_chunk
    """
    def __init__(self, shape, dtype, chunk_size=2, weeks_per_chunk=None):
        """
        Parameters
        ----------
        shape : tuple
            Array shape
        dtype : str | np.dtype
            Array data type
        chunk_size : int, optional
            Chunk size in MB, by default 2
        weeks_per_chunk : int, optional
            Number of weeks per time chunk, if None scale weeks based on 8
            weeks for hourly data, by default None
        """
        self._chunks = self.compute_dtype_chunks(
            shape, dtype,
            chunk_size=chunk_size,
            weeks_per_chunk=weeks_per_chunk)

    @staticmethod
    def _compute_time_chunk_size(t_len, weeks_per_chunk=8):
        """
        Compute the size of the temporal chunks. Chunks are based on

        Parameters
        ----------
        t_len : int
            Length of temporal axis
        weeks_per_chunk : int, optional
            Number of weeks per time chunk, if None scale weeks based on 8
            weeks for hourly data, by default None

        Returns
        -------
        time_chunks : int
            Size of chunks along temporal axis
        """
        # Freq is the number of time-steps per hour
        freq = t_len / 8760

        if weeks_per_chunk is None:
            # Use default of 8 weeks for hourly chunks
            weeks_per_chunk = ceil(8 / freq)

        # Compute time chunks
        time_chunk = ceil(weeks_per_chunk * 7 * 24 * freq)

        if time_chunk >= t_len:
            time_chunk = ceil(t_len / 4)

        return time_chunk

    @staticmethod
    def _compute_site_chunk_size(time_chunk, dtype, chunk_size=2):
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
        site_chunk = ceil(chunk_size / (time_chunk * pixel_size))

        return site_chunk

    @classmethod
    def compute_dtype_chunks(cls, shape, dtype, chunk_size=2,
                             weeks_per_chunk=None):
        """
        Compute chunks based on array shape and dtype

        Parameters
        ----------
        shape : tuple
            Array shape
        dtype : str | np.dtype
            Array data type
        chunk_size : int, optional
            Chunk size in MB, by default 2
        weeks_per_chunk : int, optional
            Number of weeks per time chunk, if None scale weeks based on 8
            weeks for hourly data, by default None

        Returns
        -------
        chunks : tuple
            Dataset chunk size along all axis
        """
        time_chunk = cls._compute_time_chunk_size(
            shape[0],
            weeks_per_chunk=weeks_per_chunk)
        site_chunk = cls._compute_site_chunk_size(time_chunk, dtype,
                                                  chunk_size=chunk_size)

        sites = np.sum(shape[1:])
        if site_chunk >= sites:
            site_chunk = shape[1:]
        elif len(shape) > 2:
            weights = [i // sites for i in shape[1:]]
            site_chunk = (ceil(w * site_chunk) for w in weights)
        else:
            site_chunk = (site_chunk, )

        chunks = (time_chunk, ) + site_chunk

        return chunks

    @classmethod
    def compute(cls, shape, dtype, chunk_size=2, weeks_per_chunk=None):
        """
        Compute chunks based on array shape and dtype

        Parameters
        ----------
        shape : tuple
            Array shape
        dtype : str | np.dtype
            Array data type
        chunk_size : int, optional
            Chunk size in MB, by default 2
        weeks_per_chunk : int, optional
            Number of weeks per time chunk, if None scale weeks based on 8
            weeks for hourly data, by default None

        Returns
        -------
        chunks : tuple
            Dataset chunk size along all axis
        """
        chunks = cls(shape, dtype, chunk_size=chunk_size,
                     weeks_per_chunk=weeks_per_chunk)

        return chunks.chunks


class ArrayChunkSize(BaseChunkSize):
    """
    Compute chunks based on array size, array will only be chunked along the
    0 axis
    """
    def __init__(self, arr, chunk_size=2):
        """
        Parameters
        ----------
        arr : ndarray
            Dataset array
        chunk_size : int, optional
            Chunk size in MB, by default 2
        """
        self._chunks = self.compute_arr_chunks(arr, chunk_size=chunk_size)

    @staticmethod
    def _get_arr_size(arr):
        """
        Get the size of the given array in MB

        Parameters
        ----------
        data : ndarray
            Data array to chunk

        Returns
        -------
        float
            Data array size in MB
        """

        return arr.size * arr.dtype.itemsize * 10**-6

    @classmethod
    def compute_arr_chunks(cls, arr, chunk_size=2):
        """
        Compute chunks based on array size, array will only be chunked along
        the 0 axis

        Parameters
        ----------
        arr : ndarray
            Dataset array
        chunk_size : int, optional
            Chunk size in MB, by default 2

        Returns
        -------
        chunks : tuple
            Dataset chunk size along all axis
        """
        arr_size = cls._get_arr_size(arr)
        if arr_size >= 2:
            arr_len = len(arr)
            chunks = ceil(arr_len // (arr_size / chunk_size))
            chunks = (chunks, ) + arr.shape[1:]
        else:
            chunks = None

        return chunks

    @classmethod
    def compute(cls, arr, chunk_size=2):
        """
        Compute chunks based on array size, array will only be chunked along
        the 0 axis

        Parameters
        ----------
        arr : ndarray
            Dataset array
        chunk_size : int, optional
            Chunk size in MB, by default 2

        Returns
        -------
        chunks : tuple
            Dataset chunk size along all axis
        """
        chunks = cls(arr, chunk_size=chunk_size)

        return chunks.chunks
