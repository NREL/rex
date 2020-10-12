# -*- coding: utf-8 -*-
"""
Execution utilities.
"""
from abc import ABC, abstractmethod
import subprocess
import logging
import getpass
import shlex
from warnings import warn

from rex.utilities.execution import SubprocessManager
from rex.utilities.exceptions import (ExecutionError, HpcError, SlurmWarning,
                                      PbsWarning)


logger = logging.getLogger(__name__)


class HpcJobManager(SubprocessManager, ABC):
    """Abstract HPC job manager framework"""

    # get username as class attribute.
    USER = getpass.getuser()

    # HPC queue column headers
    QCOL_NAME = None  # Job name column
    QCOL_ID = None  # Job integer ID column
    QCOL_STATUS = None  # Job status column

    # set a max job name length, will raise error if too long.
    MAX_NAME_LEN = 100

    # default rows to skip in queue stdout
    QSKIP = None

    def __init__(self, user=None, queue_dict=None):
        """
        Parameters
        ----------
        user : str | None
            HPC username. None will get your username using getpass.getuser()
        queue_dict : dict | None
            Parsed HPC queue dictionary (qstat for PBS or squeue for SLURM)
            from parse_queue_str(). None will get the queue from PBS or SLURM.
        """

        self._user = user
        if self._user is None:
            self._user = self.USER

        if queue_dict is not None and not isinstance(queue_dict, dict):
            emsg = ('HPC queue_dict arg must be None or Dict but received: '
                    '{}, {}'.format(queue_dict, type(queue_dict)))
            logger.error(emsg)
            raise HpcError(emsg)

        self._queue = queue_dict

    @staticmethod
    def _skip_q_rows(queue_str, skip_rows):
        """Remove rows from the queue_str that are to be skipped.

        Parameters
        ----------
        queue_str : str
            HPC queue output string. Can be split on line breaks to get list.
        skip_rows : int | list | None
            Optional row index values to skip.

        Returns
        -------
        queue_str : str
            HPC queue output string. Can be split on line breaks to get list.
        """
        if skip_rows is not None:
            if isinstance(skip_rows, int):
                skip_rows = [skip_rows]

            queue_str = [row for i, row in enumerate(queue_str.split('\n'))
                         if i not in skip_rows]
            queue_str = '\n'.join(queue_str)

        return queue_str

    @classmethod
    def parse_queue_str(cls, queue_str, keys=0):
        """Parse the qstat or squeue output string into a dict format keyed by
        integer job id with nested dictionary of job properties (queue
        printout columns).

        Parameters
        ----------
        queue_str : str
            HPC queue output string (qstat for PBS or squeue for SLURM).
            Typically a space-delimited string with line breaks.
        keys : list | int
            Argument to set the queue job attributes (column headers).
            This defaults to an integer which says which row index contains
            the space-delimited column headers. Can also be a list to
            explicitly set the column headers.

        Returns
        -------
        queue_dict : dict
            HPC queue parsed into dictionary format keyed by integer job id
            with nested dictionary of job properties (queue printout columns).
        """

        queue_dict = {}
        queue_rows = queue_str.split('\n')

        if isinstance(keys, int):
            del_index = keys
            keys = [k.strip(' ')
                    for k in queue_rows[keys].strip(' ').split(' ')
                    if k != '']
            del queue_rows[del_index]

        for row in queue_rows:
            job = [k.strip(' ') for k in row.strip(' ').split(' ') if k != '']
            job_id = int(job[keys.index(cls.QCOL_ID)])
            queue_dict[job_id] = {k: job[i] for i, k in enumerate(keys)}

        return queue_dict

    @abstractmethod
    def query_queue(self, job_name=None, user=None, qformat=None,
                    skip_rows=None):
        """Run the HPC queue command and return the raw stdout string.

        Parameters
        ----------
        job_name : str | None
            Optional to check the squeue for a specific job name (not limited
            to the 8 shown characters) or None to show user's whole queue.
        user : str | None
            HPC username. None will get your username using getpass.getuser()
        qformat : str | None
            Queue format string specification. Changing this form the
            default (None) could have adverse effects!
        skip_rows : int | list | None
            Optional row index values to skip.

        Returns
        -------
        stdout : str
            HPC queue output string. Can be split on line breaks to get list.
        """

    @property
    def queue(self):
        """Get the HPC queue parsed into dict format keyed by integer job id

        Returns
        -------
        queue : dict
            HPC queue parsed into dictionary format keyed by integer job id
            with nested dictionary of job properties (queue printout columns).
        """
        if self._queue is None:
            qstr = self.query_queue(user=self._user)
            self._queue = self.parse_queue_str(qstr)

        return self._queue

    @property
    def queue_job_names(self):
        """Get a list of the job names in the queue"""
        return [attrs[self.QCOL_NAME] for attrs in self.queue.values()]

    @property
    def queue_job_ids(self):
        """Get a list of the job integer ids in the queue"""
        return list(self.queue.keys())

    def check_status(self, job_id=None, job_name=None):
        """Check the status of an HPC job using the HPC queue.

        Parameters
        ----------
        job_id : int | None
            Job integer ID number (preferred input)
        job_name : str
            Job name string.

        Returns
        -------
        status : str | NoneType
            Queue job status str or None if not found.
            SLURM status strings: PD, R, CG (pending, running, complete).
            PBS status strings: Q, R, C (queued, running, complete).
        """

        status = None

        if job_id is not None:
            if int(job_id) in self.queue:
                status = self.queue[int(job_id)][self.QCOL_STATUS]

        elif job_name is not None:
            if job_name in self.queue_job_names:
                for attrs in self.queue.values():
                    if attrs[self.QCOL_NAME] == job_name:
                        status = attrs[self.QCOL_STATUS]
                        break

        else:
            msg = 'Need a job_id or job_name to check HPC job status!'
            logger.error(msg)
            raise HpcError(msg)

        return status


class PBS(HpcJobManager):
    """Subclass for PBS subprocess jobs."""

    # PBS qstat column headers
    QCOL_NAME = 'Name'  # Job name column
    QCOL_ID = 'Job id'  # Job integer ID column
    QCOL_STATUS = 'S'  # Job status column

    # Frozen PBS qstat column headers cached b/c its not space delimited
    QSTAT_KEYS = ('Job id', 'Name', 'User', 'Time Use', 'S', 'Queue')

    # set a max job name length, will raise error if too long.
    MAX_NAME_LEN = 100

    # default rows to skip in queue stdout
    QSKIP = (0, 1)

    def __init__(self, user=None, queue_dict=None):
        """
        Parameters
        ----------
        user : str | None
            HPC username. None will get your username using getpass.getuser()
        queue_dict : dict | None
            Parsed HPC queue dictionary (qstat for PBS or squeue for SLURM)
            from parse_queue_str(). None will get the queue from PBS or SLURM.
        """
        super().__init__(user=user, queue_dict=queue_dict)

    @classmethod
    def query_queue(cls, job_name=None, user=None, qformat=None,
                    skip_rows=None):
        """Run the PBS qstat command and return the raw stdout string.

        Parameters
        ----------
        job_name : str | None
            Optional to check the squeue for a specific job name (not limited
            to the 8 shown characters) or None to show user's whole queue.
        user : str | None
            HPC username. None will get your username using getpass.getuser()
        qformat : str | None
            Queue format string specification. Changing this form the
            default (None) could have adverse effects!
        skip_rows : int | list | None
            Optional row index values to skip.

        Returns
        -------
        stdout : str
            qstat output string. Can be split on line breaks to get list.
        """

        if user is None:
            user = cls.USER

        if skip_rows is None:
            skip_rows = cls.QSKIP

        cmd = 'qstat -u {user}'.format(user=user)
        stdout, _ = cls.submit(cmd)
        stdout = cls._skip_q_rows(stdout, skip_rows)
        return stdout

    @property
    def queue(self):
        """Get the HPC queue parsed into dict format keyed by integer job id

        Returns
        -------
        queue : dict
            HPC queue parsed into dictionary format keyed by integer job id
            with nested dictionary of job properties (queue printout columns).
        """
        if self._queue is None:
            qstr = self.query_queue(user=self._user)
            self._queue = self.parse_queue_str(qstr, keys=self.QSTAT_KEYS)

        return self._queue

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

        if len(name) > self.MAX_NAME_LEN:
            msg = ('Cannot submit job with name longer than {} chars: "{}"'
                   .format(self.MAX_NAME_LEN, name))
            logger.error(msg)
            raise ValueError(msg)

        status = self.check_status(job_name=name)

        if status in ('Q', 'R'):
            logger.info('Not submitting job "{}" because it is already in '
                        'qstat with status: "{}"'.format(name, status))
            out = None
            err = 'already_running'
        else:
            self.make_path(stdout_path)
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

            if not keep_sh:
                self.rm(fname)

            if err:
                msg = 'Received a PBS error or warning: {}'.format(err)
                logger.warning(msg)
                warn(msg, PbsWarning)
            else:
                logger.debug('PBS job "{}" with id #{} submitted successfully'
                             .format(name, out))
                self._queue[int(out)] = {self.QCOL_ID: int(out),
                                         self.QCOL_NAME: name,
                                         self.QCOL_STATUS: 'Q'}

        return out, err


class SLURM(HpcJobManager):
    """Subclass for SLURM subprocess jobs."""

    # SLURM squeue column headers
    QCOL_NAME = 'NAME'  # Job name column
    QCOL_ID = 'JOBID'  # Job integer ID column
    QCOL_STATUS = 'ST'  # Job status column

    MAX_NAME_LEN = 100
    SQ_FORMAT = ("%.15i %.30P  %.{}j  %.20u %.10t %.15M %.25R %q"
                 .format(MAX_NAME_LEN))

    # default rows to skip in queue stdout
    QSKIP = None

    def __init__(self, user=None, queue_dict=None):
        """
        Parameters
        ----------
        user : str | None
            HPC username. None will get your username using getpass.getuser()
        queue_dict : dict | None
            Parsed HPC queue dictionary (qstat for PBS or squeue for SLURM)
            from parse_queue_str(). None will get the queue from PBS or SLURM.
        """
        super().__init__(user=user, queue_dict=queue_dict)

    @classmethod
    def query_queue(cls, job_name=None, user=None, qformat=None,
                    skip_rows=None):
        """Run the HPC queue command and return the raw stdout string.

        Parameters
        ----------
        job_name : str | None
            Optional to check the squeue for a specific job name (not limited
            to the 8 shown characters) or None to show user's whole queue.
        user : str | None
            HPC username. None will get your username using getpass.getuser()
        qformat : str | None
            Queue format string specification. Changing this form the
            default (None) could have adverse effects!
        skip_rows : int | list | None
            Optional row index values to skip.

        Returns
        -------
        stdout : str
            HPC queue output string. Can be split on line breaks to get list.
        """
        job_name_str = ''
        if job_name is not None:
            job_name_str = ' -n {}'.format(job_name)

        if user is None:
            user = cls.USER

        if qformat is None:
            qformat = cls.SQ_FORMAT

        if skip_rows is None:
            skip_rows = cls.QSKIP

        cmd = ('squeue -u {user}{job_name} --format="{format_str}"'
               .format(user=user, job_name=job_name_str, format_str=qformat))
        stdout, _ = cls.submit(cmd)
        stdout = cls._skip_q_rows(stdout, skip_rows)

        return stdout

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
            self._queue = None
            for job_id in self.queue_job_ids:
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
            self._queue = None
            for job_id, attrs in self.queue.items():
                status = attrs[self.QCOL_STATUS].lower()
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
            self._queue = None
            for job_id, attrs in self.queue.items():
                status = attrs[self.QCOL_STATUS].lower()
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
            self._queue = None
            for job_id, attrs in self.queue.items():
                status = attrs[self.QCOL_STATUS].lower()
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
            sbatch standard output, if submitted successfully, this is the
            slurm job id.
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
            self.make_path(stdout_path)
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
                job_id = int(out.split(' ')[-1])
                out = str(job_id)
                logger.debug('SLURM job "{}" with id #{} submitted '
                             'successfully'.format(name, job_id))
                self._queue[job_id] = {self.QCOL_ID: job_id,
                                       self.QCOL_NAME: name,
                                       self.QCOL_STATUS: 'PD'}

        return out, err
