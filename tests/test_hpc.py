# -*- coding: utf-8 -*-
"""
pytests for hpc job managers.
"""
import shutil
import os
import pytest

from rex import TESTDATADIR
from rex.utilities.hpc import PBS, SLURM


FP_QSTAT = os.path.join(TESTDATADIR, 'hpc/qstat.txt')
with open(FP_QSTAT, 'r') as f:
    QSTAT_RAW = f.read()

FP_SQ = os.path.join(TESTDATADIR, 'hpc/squeue.txt')
with open(FP_SQ, 'r') as f:
    SQUEUE_RAW = f.read()


def test_pbs_qstat():
    """Test the PBS job handler qstat parsing utility"""
    qstat = PBS._skip_q_rows(QSTAT_RAW, PBS.QSKIP)
    qstat = PBS.parse_queue_str(qstat, keys=PBS.QSTAT_KEYS)

    ids = (1231230, 1231231, 1231232)
    assert len(qstat) == 3
    assert all([i in qstat for i in ids])

    for attrs in qstat.values():
        assert all([key in attrs for key in PBS.QSTAT_KEYS])

    assert qstat[ids[0]][PBS.QCOL_NAME] == 'my_parallel_job0'
    assert qstat[ids[0]][PBS.QCOL_STATUS] == 'R'
    assert qstat[ids[2]][PBS.QCOL_STATUS] == 'Q'


def test_pbs_check_job():
    """Test the PBS job status checker utility"""
    qstat = PBS._skip_q_rows(QSTAT_RAW, PBS.QSKIP)
    qstat = PBS.parse_queue_str(qstat, keys=PBS.QSTAT_KEYS)
    pbs = PBS(user='usr0', queue_dict=qstat)
    assert len(pbs.queue_job_names) == 3
    assert len(pbs.queue_job_ids) == 3

    s1 = pbs.check_status(job_name='my_parallel_job0')
    s2 = pbs.check_status(job_name='my_parallel_job1')
    s3 = pbs.check_status(job_id=1231232)
    s4 = pbs.check_status(job_name='bad')
    s5 = pbs.check_status(job_id=1)

    assert s1 == 'R'
    assert s2 == 'Q'
    assert s3 == 'Q'
    assert s4 is None
    assert s5 is None


def test_pbs_qsub():
    """Test the PBS job submission utility (limited test without actual
    job submission)"""
    qstat = PBS._skip_q_rows(QSTAT_RAW, PBS.QSKIP)
    qstat = PBS.parse_queue_str(qstat, keys=PBS.QSTAT_KEYS)
    pbs = PBS(user='usr0', queue_dict=qstat)
    cmd = 'python -c \"print(\'hello world\')\"'
    alloc = 'rev'
    queue = 'batch-h'
    name = 'my_parallel_job0'
    out, err = pbs.qsub(cmd, alloc, queue, name)
    assert out is None
    assert err == 'already_running'
    name = 'pbs_qsub_test'

    with pytest.raises(FileNotFoundError):
        pbs.qsub(cmd, alloc, queue, name)

    fn_sh = '{}.sh'.format(name)

    assert os.path.exists(fn_sh)
    with open(fn_sh, 'r') as f:
        sh = f.readlines()

    assert '#PBS -N {}'.format(name) in sh[1]
    assert cmd in sh[-1]

    os.remove(fn_sh)
    shutil.rmtree('stdout/')


def test_slurm_squeue():
    """Test the SLURM job handler squeue parsing utility"""
    sq = SLURM.parse_queue_str(SQUEUE_RAW)

    ids = (12345, 12346, 12347)
    assert len(sq) == 3
    assert all([i in sq for i in ids])

    req_keys = ('JOBID', 'PARTITION', 'NAME', 'USER', 'ST', 'TIME',
                'NODELIST(REASON)')
    for attrs in sq.values():
        assert all([key in attrs for key in req_keys])

    assert sq[ids[0]][SLURM.QCOL_NAME] == 'job1'
    assert sq[ids[0]][SLURM.QCOL_STATUS] == 'R'
    assert sq[ids[2]][SLURM.QCOL_STATUS] == 'PD'


def test_slurm_check_job():
    """Test the SLURM job status checker utility"""
    sq = SLURM.parse_queue_str(SQUEUE_RAW)
    slurm = SLURM(user='usr1', queue_dict=sq)
    assert len(slurm.queue_job_names) == 3
    assert len(slurm.queue_job_ids) == 3

    s1 = slurm.check_status(job_name='job1')
    s2 = slurm.check_status(job_name='job2')
    s3 = slurm.check_status(job_id=12347)
    s4 = slurm.check_status(job_name='bad')
    s5 = slurm.check_status(job_id=1)

    assert s1 == 'R'
    assert s2 == 'PD'
    assert s3 == 'PD'
    assert s4 is None
    assert s5 is None


def test_slurm_sbatch():
    """Test the SLURM job submission utility (limited test without actual
    job submission)"""
    sq = SLURM.parse_queue_str(SQUEUE_RAW)
    slurm = SLURM(user='usr1', queue_dict=sq)
    cmd = 'python -c \"print(\'hello world\')\"'
    alloc = 'rev'
    walltime = 0.43
    name = 'job1'
    out, err = slurm.sbatch(cmd, alloc, walltime, name=name)
    assert out is None
    assert err == 'already_running'
    name = 'slurm_sbatch_test'

    with pytest.raises(FileNotFoundError):
        slurm.sbatch(cmd, alloc, walltime, name=name)

    fn_sh = '{}.sh'.format(name)

    assert os.path.exists(fn_sh)
    with open(fn_sh, 'r') as f:
        sh = f.readlines()

    assert '#SBATCH --account=rev' in sh[1]
    assert '#SBATCH --time=00:26:00' in sh[2]
    assert '#SBATCH --job-name={}'.format(name) in sh[3]
    assert cmd in sh[-1]

    os.remove(fn_sh)
    shutil.rmtree('stdout/')
