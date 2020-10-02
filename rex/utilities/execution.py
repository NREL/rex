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
import getpass
import shlex
from warnings import warn

from rex.utilities.loggers import LOGGERS, log_mem
from rex.utilities.exceptions import (ExecutionError, SlurmWarning,
                                      ParallelExecutionWarning)


logger = logging.getLogger(__name__)


class SubprocessManager:
    """Base class to handle subprocess execution."""

    # get username as class attribute.
    USER = getpass.getuser()

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

        subprocess.run(cmd, shell=shell)

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


class PBS(SubprocessManager):
    """Subclass for PBS subprocess jobs."""

    def __init__(self, cmd, alloc, queue, name='reV',
                 feature=None, stdout_path='./stdout'):
        """Initialize and submit a PBS job.

        Parameters
        ----------
        cmd : str
            Command to be submitted in PBS shell script. Example:
                'python -m reV.generation.cli_gen'
        alloc : str
            HPC allocation account. Example: 'rev'.
        queue : str
            HPC queue to submit job to. Example: 'short', 'batch-h', etc...
        name : str
            PBS job name.
        feature : str | None
            PBS feature request (-l {feature}).
            Example: 'feature=24core', 'qos=high', etc...
        stdout_path : str
            Path to print .stdout and .stderr files.
        """

        self.make_path(stdout_path)
        self.id, self.err = self.qsub(cmd,
                                      alloc=alloc,
                                      queue=queue,
                                      name=name,
                                      feature=feature,
                                      stdout_path=stdout_path)

    @staticmethod
    def check_status(job_id=None, job_name=None):
        """Check the status of this PBS job using qstat.

        Parameters
        ----------
        job_id : int
            Job integer ID number.
        job_name : str
            Job name string.

        Returns
        -------
        out : str or NoneType
            Qstat job status character or None if not found.
            Common status codes: Q, R, C (queued, running, complete).
        """

        # column location of various job identifiers
        if job_id is not None:
            job = job_id
            col_loc = 0
        elif job_name is not None:
            job = job_name
            col_loc = 3
        else:
            msg = 'Need a job_id or job_name to check PBS job status!'
            logger.error(msg)
            raise ValueError(msg)

        qstat_rows = PBS.qstat()
        if qstat_rows is None:
            return None
        else:
            # reverse the list so most recent jobs are first
            qstat_rows = reversed(qstat_rows)

        # update job status from qstat list
        for row in qstat_rows:
            row = row.split()
            # make sure the row is long enough to be a job status listing
            if len(row) > 10:
                if row[col_loc].strip() == str(job).strip():
                    # Job status is located at the -2 index
                    status = row[-2]
                    logger.debug('Job "{}" has status: "{}"'
                                 .format(job, status))
                    return status
        return None

    @staticmethod
    def qstat():
        """Run the PBS qstat command and return the stdout split to rows.

        Returns
        -------
        qstat_rows : list | None
            List of strings where each string is a row in the qstat printout.
            Returns None if qstat is empty.
        """

        cmd = 'qstat -u {user}'.format(user=PBS.USER)
        stdout, _ = PBS.submit(cmd)
        if not stdout:
            # No jobs are currently running.
            return None
        else:
            qstat_rows = stdout.split('\n')
            return qstat_rows

    def qsub(self, cmd, alloc, queue, name='reV', feature=None,
             stdout_path='./stdout', keep_sh=False):
        """Submit a PBS job via qsub command and PBS shell script

        Parameters
        ----------
        cmd : str
            Command to be submitted in PBS shell script. Example:
                'python -m reV.generation.cli_gen'
        alloc : str
            HPC allocation account. Example: 'rev'.
        queue : str
            HPC queue to submit job to. Example: 'short', 'batch-h', etc...
        name : str
            PBS job name.
        feature : str | None
            PBS feature request (-l {feature}).
            Example: 'feature=24core', 'qos=high', etc...
        stdout_path : str
            Path to print .stdout and .stderr files.
        keep_sh : bool
            Boolean to keep the .sh files. Default is to remove these files
            after job submission.

        Returns
        -------
        out : str
            qsub standard output, this is typically the PBS job ID.
        err : str
            qsub standard error, this is typically an empty string if the job
            was submitted successfully.
        """

        status = self.check_status(name, var='name')

        if status in ('Q', 'R'):
            warn('Not submitting job "{}" because it is already in '
                 'qstat with status: "{}"'.format(name, status))
            out = None
            err = 'already_running'
        else:
            feature_str = '#PBS -l {}\n'.format(str(feature).replace(' ', ''))
            fname = '{}.sh'.format(name)
            script = ('#!/bin/bash\n'
                      '#PBS -N {n} # job name\n'
                      '#PBS -A {a} # allocation account\n'
                      '#PBS -q {q} # queue (debug, short, batch, or long)\n'
                      '#PBS -o {p}/{n}_$PBS_JOBID.o\n'
                      '#PBS -e {p}/{n}_$PBS_JOBID.e\n'
                      '{L}'
                      'echo Running on: $HOSTNAME, Machine Type: $MACHTYPE\n'
                      '{cmd}'
                      .format(n=name, a=alloc, q=queue, p=stdout_path,
                              L=feature_str if feature else '',
                              cmd=cmd))

            # write the shell script file and submit as qsub job
            self.make_sh(fname, script)
            out, err = self.submit('qsub {script}'.format(script=fname))

            if not err:
                logger.debug('PBS job "{}" with id #{} submitted successfully'
                             .format(name, out))
                if not keep_sh:
                    self.rm(fname)

        return out, err


class SLURM(SubprocessManager):
    """Subclass for SLURM subprocess jobs."""

    MAX_NAME_LEN = 100
    SQ_FORMAT = ("%.15i %.30P  %.{}j  %.20u %.10t %.15M %.25R %q"
                 .format(MAX_NAME_LEN))

    def __init__(self, user=None):
        """Initialize and submit a PBS job.

        Parameters
        ----------
        user : str | None
            SLURM username. None will get your username using getpass.getuser()
        """

        self._user = user
        if self._user is None:
            self._user = self.USER

        self._squeue = None
        self._submitted_job_names = []
        self._submitted_job_ids = []

    def check_status(self, job_id=None, job_name=None):
        """Check the status of this SLURM job using squeue.

        Parameters
        ----------
        job_id : int | None
            Job integer ID number (preferred input)
        job_name : str
            Job name string (not limited to displayed chars in squeue).
        var : str
            Identity/type of job identification input arg ('id' or 'name').

        Returns
        -------
        status : str | NoneType
            squeue job status str ("ST") or None if not found.
            Common status codes: PD, R, CG (pending, running, complete).
        """

        status = None
        sqd = self.squeue_dict

        if job_id is not None:
            if int(job_id) in sqd:
                status = sqd[int(job_id)]['ST']
            elif int(job_id) in self._submitted_job_ids:
                status = 'PD'

        elif job_name is not None:
            if job_name in self.squeue_job_names:
                for attrs in sqd.values():
                    if attrs['NAME'] == job_name:
                        status = attrs['ST']
                        break
            elif job_name in self._submitted_job_names:
                status = 'PD'

        else:
            msg = 'Need a job_id or job_name to check SLURM job status!'
            logger.error(msg)
            raise ValueError(msg)

        return status

    @staticmethod
    def run_squeue(job_name=None, user=None, sq_format=None):
        """Run the SLURM squeue command and return the stdout split to rows.

        Parameters
        ----------
        job_name : str | None
            Optional to check the squeue for a specific job name (not limited
            to the 8 shown characters) or show users whole squeue.
        user : str | None
            SLURM username. None will get your username using getpass.getuser()
        sq_format : str | None
            SLURM squeue format string specification. Changing this form the
            default (None) could have adverse effects!

        Returns
        -------
        stdout : str
            squeue output string. Can be split on line breaks to get list.
        """
        job_name_str = ''
        if job_name is not None:
            job_name_str = ' -n {}'.format(job_name)

        if user is None:
            user = SLURM.USER

        if sq_format is None:
            sq_format = SLURM.SQ_FORMAT

        cmd = ('squeue -u {user}{job_name} --format="{format_str}"'
               .format(user=user, job_name=job_name_str, format_str=sq_format))
        stdout, _ = SLURM.submit(cmd)

        return stdout

    @property
    def squeue(self):
        """Get the cached squeue output string (no special formatting)"""
        if self._squeue is None:
            self._squeue = self.run_squeue(user=self._user)
        return self._squeue

    @property
    def squeue_dict(self):
        """Get the squeue output as a dict keyed by integer job id with nested
        dictionary of squeue job properties (columns)."""

        sq = self.squeue.split('\n')
        sqd = {}
        keys = [k.strip(' ') for k in sq[0].strip(' ').split(' ') if k != '']
        for row in sq[1:]:
            job = [k.strip(' ') for k in row.strip(' ').split(' ') if k != '']
            sqd[int(job[0])] = {k: job[i] for i, k in enumerate(keys)}

        return sqd

    @property
    def squeue_job_names(self):
        """Get a list of the job names in the squeue output"""
        names = []
        if self.squeue_dict:
            names = [attrs['NAME'] for attrs in self.squeue_dict.values()]
        return names

    @staticmethod
    def scontrol(cmd):
        """Submit an scontrol command.

        Parameters
        ----------
        cmd : str
            Command string after "scontrol" word
        """
        cmd = 'scontrol {}'.format(cmd)
        cmd = shlex.split(cmd)
        subprocess.call(cmd)

    def scancel(self, arg):
        """Cancel a slurm job.

        Parameters
        ----------
        arg : int | list | str
            SLURM integer job id(s) to cancel. Can be a list of integer
            job ids, 'all' to cancel all jobs, or a feature (-p short) to
            cancel all jobs with a given feature
        """

        if isinstance(arg, (list, tuple)):
            for job_id in arg:
                self.scancel(job_id)

        elif str(arg).lower() == 'all':
            self._squeue = None
            for job_id in self.squeue_dict.keys():  # pylint: disable=C0201
                self.scancel(job_id)

        elif isinstance(arg, (int, str)):
            cmd = ('scancel {}'.format(arg))
            cmd = shlex.split(cmd)
            subprocess.call(cmd)

        else:
            e = ('Could not cancel: {} with type {}'
                 .format(arg, type(arg)))
            logger.error(e)
            raise ExecutionError(e)

    def change_qos(self, arg, qos):
        """Change the priority (quality of service) for a job.

        Parameters
        ----------
        arg : int | list | str
            SLURM integer job id(s) to change qos for.
            Can be 'all' for all jobs.
        qos : str
            New qos value
        """

        if isinstance(arg, (list, tuple)):
            for job_id in arg:
                self.change_qos(job_id, qos)

        elif isinstance(arg, int):
            cmd = 'update job {} QOS={}'.format(arg, qos)
            self.scontrol(cmd)

        elif str(arg).lower() == 'all':
            self._squeue = None
            for job_id, attrs in self.squeue_dict.items():
                status = attrs['ST'].lower()
                if status == 'pd':
                    self.change_qos(job_id, qos)

        else:
            e = ('Could not change qos of: {} with type {}'
                 .format(arg, type(arg)))
            logger.error(e)
            raise ExecutionError(e)

    def hold(self, arg):
        """Temporarily hold a job from submitting. Held jobs will stay in queue
        but will not get nodes until released.

        Parameters
        ----------
        arg : int | list | str
            SLURM integer job id(s) to hold. Can be 'all' to hold all jobs.
        """

        if isinstance(arg, (list, tuple)):
            for job_id in arg:
                self.hold(job_id)

        elif isinstance(arg, int):
            cmd = 'hold {}'.format(arg)
            self.scontrol(cmd)

        elif str(arg).lower() == 'all':
            self._squeue = None
            for job_id, attrs in self.squeue_dict.items():
                status = attrs['ST'].lower()
                if status == 'pd':
                    self.hold(job_id)

        else:
            e = ('Could not hold: {} with type {}'
                 .format(arg, type(arg)))
            logger.error(e)
            raise ExecutionError(e)

    def release(self, arg):
        """Release a job that was previously on hold so it will be submitted
        to a compute node.

        Parameters
        ----------
        arg : int | list | str
            SLURM integer job id(s) to release.
            Can be 'all' to release all jobs.
        """

        if isinstance(arg, (list, tuple)):
            for job_id in arg:
                self.release(job_id)

        elif isinstance(arg, int):
            cmd = 'release {}'.format(arg)
            self.scontrol(cmd)

        elif str(arg).lower() == 'all':
            self._squeue = None
            for job_id, attrs in self.squeue_dict.items():
                status = attrs['ST'].lower()
                reason = attrs['NODELIST(REASON)'].lower()
                if status == 'pd' and 'jobheld' in reason:
                    self.release(job_id)

        else:
            e = ('Could not release: {} with type {}'
                 .format(arg, type(arg)))
            logger.error(e)
            raise ExecutionError(e)

    @staticmethod
    def _special_cmd_strs(feature, memory, module, module_root, conda_env):
        """Get special sbatch request strings for SLURM features, memory,
        modules, and conda environments

        Parameters
        ----------
        feature : str
            Additional flags for SLURM job. Format is "--qos=high"
            or "--depend=[state:job_id]". Default is None.
        memory : int
            Node memory request in GB.
        module : bool
            Module to load
        module_root : str
            Path to module root to load
        conda_env : str
            Conda environment to activate

        Returns
        -------
        feature_str : str
            SBATCH shell script feature request string.
        mem_str : str
            SBATCH shell script memory request string.
        env_str : str
            SBATCH shell script module load or source activate environment
            request string.
        """
        feature_str = ''
        if feature is not None:
            feature_str = '#SBATCH {}  # extra feature\n'.format(feature)

        mem_str = ''
        if memory is not None:
            mem_str = ('#SBATCH --mem={}  # node RAM in MB\n'
                       .format(int(memory * 1000)))

        env_str = ''
        if module is not None:
            env_str = ("echo module use {module_root}\n"
                       "module use {module_root}\n"
                       "echo module load {module}\n"
                       "module load {module}\n"
                       "echo module load complete!\n"
                       .format(module_root=module_root, module=module))
        elif conda_env is not None:
            env_str = ("echo source activate {conda_env}\n"
                       "source activate {conda_env}\n"
                       "echo conda env activate complete!\n"
                       .format(conda_env=conda_env))

        return feature_str, mem_str, env_str

    def sbatch(self, cmd, alloc, walltime, memory=None, feature=None,
               name='reV', stdout_path='./stdout', keep_sh=False,
               conda_env=None, module=None,
               module_root='/shared-projects/rev/modulefiles'):
        """Submit a SLURM job via sbatch command and SLURM shell script

        Parameters
        ----------
        cmd : str
            Command to be submitted in SLURM shell script. Example:
                'python -m reV.generation.cli_gen'
        alloc : str
            HPC project (allocation) handle. Example: 'rev'.
        walltime : float
            Node walltime request in hours.
        memory : int
            Node memory request in GB.
        feature : str
            Additional flags for SLURM job. Format is "--qos=high"
            or "--depend=[state:job_id]". Default is None.
        name : str
            SLURM job name.
        stdout_path : str
            Path to print .stdout and .stderr files.
        keep_sh : bool
            Boolean to keep the .sh files. Default is to remove these files
            after job submission.
        conda_env : str
            Conda environment to activate
        module : bool
            Module to load
        module_root : str
            Path to module root to load

        Returns
        -------
        out : str
            sbatch standard output, this is typically the SLURM job ID.
        err : str
            sbatch standard error, this is typically an empty string if the job
            was submitted successfully.
        """

        if len(name) > self.MAX_NAME_LEN:
            msg = ('Cannot submit job with name longer than {} chars: "{}"'
                   .format(self.MAX_NAME_LEN, name))
            logger.error(msg)
            raise ValueError(msg)

        status = self.check_status(job_name=name)

        if status is not None:
            logger.info('Not submitting job "{}" because it is in '
                        'squeue or has been recently submitted'.format(name))
            out = None
            err = 'already_running'

        else:
            special = self._special_cmd_strs(feature, memory, module,
                                             module_root, conda_env)
            fname = '{}.sh'.format(name)
            script = ('#!/bin/bash\n'
                      '#SBATCH --account={a}  # allocation account\n'
                      '#SBATCH --time={t}  # walltime\n'
                      '#SBATCH --job-name={n}  # job name\n'
                      '#SBATCH --nodes=1  # number of nodes\n'
                      '#SBATCH --output={p}/{n}_%j.o\n'
                      '#SBATCH --error={p}/{n}_%j.e\n{m}{f}'
                      'echo Running on: $HOSTNAME, Machine Type: $MACHTYPE\n'
                      '{e}\n{cmd}'
                      .format(a=alloc, t=self.format_walltime(walltime),
                              n=name, p=stdout_path, m=special[1],
                              f=special[0], e=special[2], cmd=cmd))

            # write the shell script file and submit as qsub job
            self.make_sh(fname, script)
            out, err = self.submit('sbatch {script}'.format(script=fname))

            if not keep_sh:
                self.rm(fname)

            if err:
                msg = 'Received a SLURM error or warning: {}'.format(err)
                logger.warning(msg)
                warn(msg, SlurmWarning)
            else:
                logger.debug('SLURM job "{}" with id #{} submitted '
                             'successfully'.format(name, out))
                self._submitted_job_names.append(name)
                try:
                    self._submitted_job_ids.append(int(out))
                except ValueError:
                    msg = ('SLURM sbatch output for "{}" was not job id! '
                           'sbatch stdout: {} sbatch stderr: {}'
                           .format(name, out, err))
                    logger.error(msg)
                    raise ValueError(msg)

        return out, err


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
            kwargs['initializer'] = LOGGERS.init_logger
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
