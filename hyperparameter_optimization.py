from ax.core.base_trial import TrialStatus
import submitit
from typing_extensions import NamedTuple
from time import sleep
from os import listdir
#import logging
#submitit.logger.setLevel(logging.DEBUG)

class SlurmJob(NamedTuple):
    """Represent a job scheduled"""

    id: int
    parameters: dict


def test(params):
    import socket
    print(socket.gethostname())
    listdir('/')
    sleep(10)
    import socket
    print(socket.gethostname())

    return params["a"] * params["b"]


class SlurmJobQueueClient:
    def schedule_job(self, parameters):
        log_folder = "log_run/%j"
        running_dir = "/home/pml11/github_pml/"
        executor = submitit.AutoExecutor(folder=log_folder, cluster="slurm", slurm_python="/usr/bin/apptainer run --nv --env-file .env --bind /home/space/datasets:/home/space/datasets pml.sif python")
        executor.update_parameters(
            slurm_partition="cpu-2h",
            #slurm_gpus_per_node=1,
            slurm_cpus_per_task=4,
            slurm_job_name="hyper_param_opt",
            slurm_additional_parameters={"chdir": running_dir, "export": "ALL,PATH=$PATH:/usr/bin/python3:/opt/slurm/bin"},

        )
        print("We are here")
        job = executor.submit(test, parameters)
        print(job.job_id)
        print(job._tasks)
        print(job.result())
        print(job.paths)
        output = 5
        print(output)


import socket
print(socket.gethostname())

client = SlurmJobQueueClient()
client.schedule_job({"a": 5, "b": 6})
# SBATCH --job-name=hyper
# SBATCH --partition=gpu-teaching-2h # Run on the 2h GPU runtime partition, also 5h, 2d and 7d availabl
# SBATCH --gpus-per-node=1 # One A100 with 40GB
# SBATCH --ntasks-per-node=4 # 4 CPU threads
# SBATCH --output=/home/pml11/github_pml/output.txt
# SBATCH --error=/home/pml11/github_pml/error.txt
# SBATCH --chdir=/home/pml11/github_pml
# SBATCH --array=1-2 # run ten times
