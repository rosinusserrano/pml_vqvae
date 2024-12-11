from ax import optimize
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.utils.notebook.plotting import render
from ax.service.utils.report_utils import exp_to_df
from submitit import AutoExecutor, LocalJob, DebugJob

import submitit
from typing_extensions import NamedTuple
from time import sleep
from os import listdir

# import logging
# submitit.logger.setLevel(logging.DEBUG)


class SlurmJob(NamedTuple):
    """Represent a job scheduled"""

    id: int
    parameters: dict


def test(params):
    import socket

    print(socket.gethostname())
    listdir("/")
    sleep(10)
    import socket

    print(socket.gethostname())

    return params["a"] * params["b"]


class SlurmJobQueueClient:
    def schedule_job(self, parameters):
        log_folder = "log_run/%j"
        running_dir = "/home/pml11/github_pml/"
        executor = submitit.AutoExecutor(
            folder=log_folder,
            cluster="slurm",
            slurm_python="/usr/bin/apptainer run --nv --env-file .env --bind /home/space/datasets:/home/space/datasets pml.sif python",
        )
        executor.update_parameters(
            slurm_partition="cpu-2h",
            # slurm_gpus_per_node=1,
            slurm_cpus_per_task=4,
            slurm_job_name="hyper_param_opt",
            slurm_additional_parameters={
                "chdir": running_dir,
                "export": "ALL,PATH=$PATH:/usr/bin/python3:/opt/slurm/bin",
            },
        )
        print("We are here")
        job = executor.submit(test, parameters)
        print(job.job_id)
        print(job._tasks)
        print(job.result())
        print(job.paths)
        output = 5
        print(output)


"""• beta in loss parameter
• PixelCNN parameters (tdb)
2.2 Training Hyperparameters
• epochs/early stoppage
• batch size"""


def evaluate(parameters):
    return parameters["beta_discrete_code_commitment"] * parameters["learning_rate"]


ax_client = AxClient()
ax_client.create_experiment(
    name="hyper_param_optimization",
    parameters=[
        {
            "name": "hidden_dimensions",
            "type": "choice",
            "values": [16, 32, 64, 128, 256, 512],
            "sort_values": True,
            "is_ordered": True,
        },
        {
            "name": "codebook_size",
            "type": "choice",
            "values": [64, 128, 256, 512, 1024, 2048],
            "sort_values": True,
            "is_ordered": True,
        },
        {
            "name": "beta_discrete_code_commitment",
            "type": "range",
            "bounds": [0.01, 2.0],
        },
        {
            "name": "optimizer",
            "type": "choice",
            "values": ["adam", "stochastic_gd", "adagrad"],
            "is_ordered": False,
            "sort_values": False,
        },
        {
            "name": "learning_rate",
            "type": "range",
            "bounds": [1e-6, 0.1],
            "log_scale": True,
        },
        {
            "name": "batch_size",
            "type": "choice",
            "values": [32, 64, 128, 256, 512, 1024, 2048],
            "sort_values": True,
            "is_ordered": True,
        },
    ],
    objectives={"mse": ObjectiveProperties(minimize=True)},
)
print(ax_client.get_max_parallelism())

for i in range(50):
    parametrization, trial_index = ax_client.get_next_trial()
    if trial_index is not None:
        ax_client.complete_trial(
            trial_index=trial_index, raw_data=evaluate(parametrization)
        )
best, values = ax_client.get_best_parameters()
render(
    ax_client.get_contour_plot(
        param_x="learning_rate", param_y="beta_discrete_code_commitment"
    )
)
print(best, values)
"""
import socket
print(socket.gethostname())

client = SlurmJobQueueClient()
client.schedule_job({"a": 5, "b": 6})
"""
