# -*- coding: utf-8 -*-
"""
Execution utilities.
"""
import multiprocessing
import concurrent.futures as cf
import subprocess
import logging
import gc
from math import floor
import os
import psutil
import shlex
from warnings import warn

from rex.utilities.loggers import LOGGERS, log_mem
from rex.utilities.exceptions import ExecutionError, ParallelExecutionWarning


logger = logging.getLogger(__name__)


class SubprocessManager:
    """Base class to handle subprocess execution."""

    @staticmethod
    def make_path(d):
        """Make a directory tree if it doesn't exist.

        Parameters
        ----------
        d : str
            Directory tree to check and potentially create.
        """
        if not os.path.exists(d):
            os.makedirs(d)

    @staticmethod
    def make_sh(fname, script):
        """Make a shell script (.sh file) to execute a subprocess.

        Parameters
        ----------
        fname : str
            Name of the .sh file to create.
        script : str
            Contents to be written into the .sh file.
        """
        logger.debug('The shell script "{n}" contains the following:\n'
                     '~~~~~~~~~~ {n} ~~~~~~~~~~\n'
                     '{s}\n'
                     '~~~~~~~~~~ {n} ~~~~~~~~~~'
                     .format(n=fname, s=script))
        with open(fname, 'w+') as f:
            f.write(script)

    @staticmethod
    def rm(fname):
        """Remove a file.

        Parameters
        ----------
        fname : str
            Filename (with path) to remove.
        """
        os.remove(fname)

    @staticmethod
    def _subproc_popen(cmd):
        """Open a subprocess popen constructor and submit a command.

        Parameters
        ----------
        cmd : str
            Command to be submitted using python subprocess.

        Returns
        -------
        stdout : str
            Subprocess standard output. This is decoded from the subprocess
            stdout with rstrip.
        stderr : str
            Subprocess standard error. This is decoded from the subprocess
            stderr with rstrip. After decoding/rstrip, this will be empty if
            the subprocess doesn't return an error.
        """

        cmd = shlex.split(cmd)

        # use subprocess to submit command and get piped o/e
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        stderr = stderr.decode('ascii').rstrip()
        stdout = stdout.decode('ascii').rstrip()

        if process.returncode != 0:
            raise OSError('Subprocess submission failed with return code {} '
                          'and stderr:\n{}'
                          .format(process.returncode, stderr))

        return stdout, stderr

    @staticmethod
    def _subproc_run(cmd, background=False, background_stdout=False,
                     shell=True):
        """Open a subprocess and submit a command.

        Parameters
        ----------
        cmd : str
            Command to be submitted using python subprocess.
        background : bool
            Flag to submit subprocess in the background. stdout stderr will
            be empty strings if this is True.
        background_stdout : bool
            Flag to capture the stdout/stderr from the background process
            in a nohup.out file.
        """
        nohup_cmd = None

        if background and background_stdout:
            nohup_cmd = 'nohup {} &'

        elif background and not background_stdout:
            nohup_cmd = 'nohup {} </dev/null >/dev/null 2>&1 &'

        if nohup_cmd is not None:
            cmd = nohup_cmd.format(cmd)
            shell = True

        subprocess.run(cmd, shell=shell, check=True)

    @staticmethod
    def submit(cmd, background=False, background_stdout=False):
        """Open a subprocess and submit a command.

        Parameters
        ----------
        cmd : str
            Command to be submitted using python subprocess.
        background : bool
            Flag to submit subprocess in the background. stdout stderr will
            be empty strings if this is True.
        background_stdout : bool
            Flag to capture the stdout/stderr from the background process
            in a nohup.out file.

        Returns
        -------
        stdout : str
            Subprocess standard output. This is decoded from the subprocess
            stdout with rstrip.
        stderr : str
            Subprocess standard error. This is decoded from the subprocess
            stderr with rstrip. After decoding/rstrip, this will be empty if
            the subprocess doesn't return an error.
        """

        if background:
            SubprocessManager._subproc_run(
                cmd, background=background,
                background_stdout=background_stdout)
            stdout, stderr = '', ''

        else:
            stdout, stderr = SubprocessManager._subproc_popen(cmd)

        return stdout, stderr

    @staticmethod
    def s(s):
        """Format input as str w/ appropriate quote types for python cli entry.

        Examples
        --------
            list, tuple -> "['one', 'two']"
            dict -> "{'key': 'val'}"
            int, float, None -> '0'
            str, other -> 'string'
        """

        if isinstance(s, (list, tuple, dict)):
            return '"{}"'.format(s)
        elif not isinstance(s, (int, float, type(None))):
            return "'{}'".format(s)
        else:
            return '{}'.format(s)

    @staticmethod
    def format_walltime(hours):
        """Get the SLURM walltime string in format "HH:MM:SS"

        Parameters
        ----------
        hours : float | int
            Requested number of job hours.

        Returns
        -------
        walltime : str
            SLURM walltime request in format "HH:MM:SS"
        """

        m_str = '{0:02d}'.format(round(60 * (hours % 1)))
        h_str = '{0:02d}'.format(floor(hours))
        return '{}:{}:00'.format(h_str, m_str)


class SpawnProcessPool(cf.ProcessPoolExecutor):
    """An adaptation of concurrent futures ProcessPoolExecutor with
    spawn processes instead of fork or forkserver."""

    def __init__(self, *args, loggers=None, **kwargs):
        """
        Parameters
        ----------
        loggers : str | list, optional
            logger(s) to initialize on workers, by default None
        """
        if 'mp_context' in kwargs:
            w = ('SpawnProcessPool being initialized with mp_context: "{}". '
                 'This will override default SpawnProcessPool behavior.'
                 .format(kwargs['mp_context']))
            logger.warning(w)
            warn(w, ParallelExecutionWarning)
        else:
            kwargs['mp_context'] = multiprocessing.get_context('spawn')

        if loggers is not None:
            kwargs['initializer'] = LOGGERS.init_loggers
            kwargs['initargs'] = (loggers, )

        super().__init__(*args, **kwargs)


def execute_parallel(fun, execution_iter, n_workers=None, **kwargs):
    """Execute concurrent futures with an established cluster.

    Parameters
    ----------
    fun : function
        Python function object that will be submitted to futures. See
        downstream execution methods for arg passing structure.
    execution_iter : iter
        Python iterator that controls the futures submitted in parallel.
    n_workers : int
        Number of workers to run in parallel
    **kwargs : dict
        Key word arguments passed to the fun.

    Returns
    -------
    results : list
        List of futures results.
    """
    futures = []
    # initialize a client based on the input cluster.
    with SpawnProcessPool(max_workers=n_workers) as executor:

        # iterate through split executions, submitting each to worker
        for i, exec_slice in enumerate(execution_iter):
            logger.debug('Kicking off serial worker #{} for: {}'
                         .format(i, exec_slice))
            # submit executions and append to futures list
            futures.append(executor.submit(execute_single, fun, exec_slice,
                                           worker=i, **kwargs))

        # gather results
        results = [future.result() for future in futures]

    return results


def execute_single(fun, input_obj, worker=0, **kwargs):
    """Execute a serial compute on a single core.

    Parameters
    ----------
    fun : function
        Function to execute.
    input_obj : object
        Object passed as first argument to fun. Typically a project control
        object that can be the result of iteration in the parallel execution
        framework.
    worker : int
        Worker number (for debugging purposes).
    **kwargs : dict
        Key word arguments passed to fun.
    """

    logger.debug('Running single serial execution on worker #{} for: {}'
                 .format(worker, input_obj))
    out = fun(input_obj, **kwargs)
    log_mem()

    return out


class SmartParallelJob:
    """Single node parallel compute manager with smart data flushing."""

    def __init__(self, obj, execution_iter, n_workers=None, mem_util_lim=0.7):
        """Single node parallel compute manager with smart data flushing.

        Parameters
        ----------
        obj : object
            Python object that will be submitted to futures. Must have methods
            run(arg) and flush(). run(arg) must take the iteration result of
            execution_iter as the single positional argument. Additionally,
            the results of obj.run(arg) will be pa ssed to obj.out. obj.out
            will be passed None when the memory is to be cleared. It is
            advisable that obj.run() be a @staticmethod for dramatically
            faster submission in parallel.
        execution_iter : iter
            Python iterator that controls the futures submitted in parallel.
        n_workers : int
            Number of workers to use in parallel. None will use all
            available workers.
        mem_util_lim : float
            Memory utilization limit (fractional). If the used memory divided
            by the total memory is greater than this value, the obj.out will
            be flushed and the local node memory will be cleared.
        """

        if not hasattr(obj, 'run') or not hasattr(obj, 'flush'):
            raise ExecutionError('Parallel execution with object: "{}" '
                                 'failed. The target object must have methods '
                                 'run() and flush()'.format(obj))
        self._obj = obj
        self._execution_iter = execution_iter
        self._n_workers = n_workers
        self._mem_util_lim = mem_util_lim

    @property
    def execution_iter(self):
        """Get the iterator object that controls the parallel execution.

        Returns
        -------
        _execution_iter : iterable
            Iterable object that controls the processes of the parallel job.
        """
        return self._execution_iter

    @property
    def mem_util_lim(self):
        """Get the memory utilization limit (fractional).

        Returns
        -------
        _mem_util_lim : float
            Fractional memory utilization limit. If the used memory divided
            by the total memory is greater than this value, the obj.out will
            be flushed and the local node memory will be cleared.
        """
        return self._mem_util_lim

    @property
    def n_workers(self):
        """Get the number of workers in the local cluster.

        Returns
        -------
        _n_workers : int
            Number of workers. Default value is the number of CPU's.
        """
        if self._n_workers is None:
            self._n_workers = os.cpu_count()

        return self._n_workers

    @property
    def obj(self):
        """Get the main python object that will be submitted to futures.

        Returns
        -------
        _obj : Object
            Python object that will be submitted to futures. Must have methods
            run(arg) and flush(). run(arg) must take the iteration result of
            execution_iter as the single positional argument. Additionally,
            the results of obj.run(arg) will be passed to obj.out. obj.out
            will be passed None when the memory is to be cleared. It is
            advisable that obj.run() be a @staticmethod for dramatically
            faster submission in parallel.
        """
        return self._obj

    def flush(self):
        """Flush obj.out to disk, set obj.out=None, and garbage collect."""
        # memory utilization limit exceeded, flush memory to disk
        self.obj.flush()
        self.obj.out = None
        gc.collect()

    def gather_and_flush(self, i, futures, force_flush=False):
        """Wait on futures, potentially update obj.out and flush to disk.

        Parameters
        ----------
        i : int | str
            Iteration number (for logging purposes).
        futures : list
            List of parallel future objects to wait on or gather.
        force_flush : bool
            Option to force a disk flush. Useful for end-of-iteration. If this
            is False, will only flush to disk if the memory utilization exceeds
            the mem_util_lim.

        Returns
        -------
        futures : list
            List of parallel future objects. If the memory was flushed, this is
            a cleared list: futures.clear()
        """

        # gather on each iteration so there is no big mem spike during flush
        # (obj.out should be a property setter that will append new data.)
        self.obj.out = [future.result() for future in futures]
        futures.clear()

        # useful log statements
        mem = psutil.virtual_memory()
        logger.info('Parallel run at iteration {0}. '
                    'Memory utilization is {1:.3f} GB out of {2:.3f} GB '
                    'total ({3:.1f}% used, limit of {4:.1f}%)'
                    .format(i, mem.used / 1e9, mem.total / 1e9,
                            100 * mem.used / mem.total,
                            100 * self.mem_util_lim))

        # check memory utilization against the limit
        if ((mem.used / mem.total) >= self.mem_util_lim) or force_flush:

            # restart client to free up memory
            # also seems to sync stderr messages (including warnings)
            # flush data to disk
            logger.info('Flushing memory to disk. The memory utilization is '
                        '{0:.2f}% and the limit is {1:.2f}%.'
                        .format(100 * (mem.used / mem.total),
                                100 * self.mem_util_lim))
            self.flush()

        return futures

    def run(self, **kwargs):
        """
        Run ParallelSmartJobs

        Parameters
        ----------
        kwargs : dict
            Keyword arguments to be passed to obj.run(). Makes it easier to
            have obj.run() as a @staticmethod.
        """

        logger.info('Executing parallel run on a local cluster with '
                    '{0} workers over {1} total iterations.'
                    .format(self.n_workers, 1 + len(self.execution_iter)))
        log_mem()

        # initialize a client based on the input cluster.
        with SpawnProcessPool(max_workers=self.n_workers) as executor:
            futures = []

            # iterate through split executions, submitting each to worker
            for i, exec_slice in enumerate(self.execution_iter):
                logger.debug('Kicking off serial worker #{0} for: {1}. '
                             .format(i, exec_slice))

                # submit executions and append to futures list
                futures.append(executor.submit(self.obj.run, exec_slice,
                                               **kwargs))

                # Take a pause after one complete set of workers
                if (i + 1) % self.n_workers == 0:
                    futures = self.gather_and_flush(i, futures)

            # All futures complete
            self.gather_and_flush('END', futures, force_flush=True)
            logger.debug('Smart parallel job complete. Returning execution '
                         'control to higher level processes.')
            log_mem()

    @classmethod
    def execute(cls, obj, execution_iter, n_workers=None,
                mem_util_lim=0.7, **kwargs):
        """Execute the smart parallel run with data flushing.

        Parameters
        ----------
        obj : object
            Python object that will be submitted to futures. Must have methods
            run(arg) and flush(). run(arg) must take the iteration result of
            execution_iter as the single positional argument. Additionally,
            the results of obj.run(arg) will be passed to obj.out. obj.out
            will be passed None when the memory is to be cleared. It is
            advisable that obj.run() be a @staticmethod for dramatically
            faster submission in parallel.
        execution_iter : iter
            Python iterator that controls the futures submitted in parallel.
        n_workers : int
            Number of workers to scale the cluster to. None will use all
            available workers in a local cluster.
        mem_util_lim : float
            Memory utilization limit (fractional). If the used memory divided
            by the total memory is greater than this value, the obj.out will
            be flushed and the local node memory will be cleared.
        kwargs : dict
            Keyword arguments to be passed to obj.run(). Makes it easier to
            have obj.run() as a @staticmethod.
        """

        manager = cls(obj, execution_iter, n_workers=n_workers,
                      mem_util_lim=mem_util_lim)
        manager.run(**kwargs)
