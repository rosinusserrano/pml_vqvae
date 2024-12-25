"""to run: apptainer run --env-file .env --nv --bind /home/space/,/etc/slurm,/opt/slurm,/opt/slurm-23.2,/etc/munge,
/var/run/munge,/usr/lib/x86_64-linux-gnu/libmunge.so.2 pml.sif python hyperparameter_optimization.py"""

from inspect import get_annotations
import warnings
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.service.utils.report_utils import exp_to_df
import submitit
from time import sleep
import random

from pml_vqvae.train_config import TrainConfig
import pml_vqvae.train

warnings.simplefilter(action="ignore", category=FutureWarning)


EXPERIMENT_NAME = "hyperopt-III-adaptive-number-of-epochs"


def get_adaptive_number_of_epochs(
    lr: float,
    bs: int,
    n: int,
    total_length: float = 1,
    max_epochs: int = 20,
):
    """Return the number of epochs that result in a unified total length.

    Reason for this is that when running the hyperoptimization with a fixed
    number of epochs we will probably favor high learning rates or small batch
    sizes, simply because they will result in either larger optimization steps
    or a higher number of iterations per epoch respectively. Therefore I
    implemented this "adaptive" procedure which should result in a number of
    epochs that hopefully will result in outcomes that can be better compared
    to each other. This is obviously not perfect, as it doesnt account for
    different behaviour of gradients when changing batch size, but is at least
    a start.

    We compute the total path length of the optimization as follows:

    total_length = learning_rate * (dataset_size / batch_size) * n_epochs

    From which results the adaptive number of epochs:

    adaptive_n_epochs = total_length / (learning_rate * (dataset_size / batch_size))
    """

    return min(max_epochs, max(1, int(total_length / (lr * (n / bs)))))


FIXED_HYPERPARAMS = {
    "dataset": "imagenet",
    "experiment_name": EXPERIMENT_NAME,
    "model_name": "vqvae",
    "n_test": 5000,
    "n_train": 25000,
    "test_interval": 1,
    "vis_train_interval": 1,
}


VQVAE_HYPERPARAMETER_SEARCH_SPACE = [
    {
        "name": "hidden_dimension",
        "type": "choice",
        "values": [16, 32, 64, 128, 256],
        "sort_values": True,
        "is_ordered": True,
    },
    {
        "name": "embedding_dimension",
        "type": "choice",
        "values": [16, 32, 64, 128, 256, 512],
        "sort_values": True,
        "is_ordered": True,
    },
    {
        "name": "codebook_size",
        "type": "choice",
        "values": [64, 256, 512, 1024],
        "sort_values": True,
        "is_ordered": True,
    },
    {
        "name": "commitment_weight",
        "type": "range",
        "bounds": [0.01, 10.0],
    },
]

PIXELCNN_HYPERPARAMETER_SEARCH_SPACE = [
    {
        "name": "hidden_chan",
        "type": "choice",
        "values": [16, 32, 64, 128, 256],
        "sort_values": True,
        "is_ordered": True,
    },
    {
        "name": "dilations",
        "type": "choice",
        "values": [
            [1, 2, 1, 4, 1, 2, 1],
            [1, 2, 1, 4, 1, 2, 1, 2, 1],
            [1, 2, 1, 3, 1, 4, 1, 3, 1, 2, 1],
            [1, 1, 2, 2, 3, 3, 4, 4],
        ],
        "sort_values": False,
        "is_ordered": False,
    },
]


TRAINING_HYPERPARAMETER_SEARCH_SPACE = [
    {
        "name": "optimizer",
        "type": "choice",
        "values": ["adam", "adamax"],
        "is_ordered": False,
        "sort_values": False,
    },
    {
        "name": "learning_rate",
        "type": "range",
        "bounds": [1e-6, 1e-1],
        "log_scale": True,
    },
    {
        "name": "batch_size",
        "type": "choice",
        "values": [32, 64, 128, 256, 512],
        "sort_values": True,
        "is_ordered": True,
    },
    {
        "name": "momentum",
        "type": "range",
        "bounds": [0.0, 0.999],
    },
    {
        "name": "weight_decay",
        "type": "range",
        "bounds": [1e-6, 1e-1],
        "log_scale": True,
    },
]


def test(parameters):
    train_config = {}
    model_config = {}

    for k, v in parameters.items():
        if k in get_annotations(TrainConfig).keys():
            train_config[k] = v
        else:
            model_config[k] = v

    train_config["model_config"] = model_config

    train_config = train_config | FIXED_HYPERPARAMS

    train_config["epochs"] = get_adaptive_number_of_epochs(
        lr=train_config["learning_rate"],
        n=train_config["n_train"],
        bs=train_config["batch_size"],
        total_length=1,
        max_epochs=20,
    )

    train_config = TrainConfig.from_dict(train_config)

    last_average_test_loss = pml_vqvae.train.train(train_config)
    # last_average_test_loss = 2

    return last_average_test_loss


class SlurmJobQueueClient:
    def __init__(self):
        log_folder = "log_run/%j"
        running_dir = "/home/pml10/pml_vqvae/"
        self.training_executor = submitit.AutoExecutor(
            folder=log_folder,
            cluster="slurm",
            slurm_python="/usr/bin/apptainer run --nv --env-file .env --bind "
            "/home/space/datasets:/home/space/datasets pml.sif python",
        )
        self.training_executor.update_parameters(
            slurm_partition="gpu-teaching-5h",
            slurm_gpus_per_node=1,
            slurm_cpus_per_task=1,
            timeout_min=300,
            slurm_job_name="hyper_param_opt",
            slurm_additional_parameters={
                "chdir": running_dir,
            },
        )

    def submit_training_job(self, parameters):
        try:
            job = self.training_executor.submit(test, parameters)
        except Exception as e:
            print(e)
            return False
        return job


def main():
    ax_client = AxClient()
    ax_client.create_experiment(
        name=EXPERIMENT_NAME,
        parameters=TRAINING_HYPERPARAMETER_SEARCH_SPACE
        + VQVAE_HYPERPARAMETER_SEARCH_SPACE,
        objectives={"mse": ObjectiveProperties(minimize=True)},
    )

    slurm_queue_client = SlurmJobQueueClient()

    total_budget = 100
    num_parallel_jobs = 2
    active_jobs = []
    submitted_jobs = 0

    while submitted_jobs < total_budget or active_jobs:
        for job, trial_index in active_jobs[:]:
            if job.done():
                result = job.result()
                ax_client.complete_trial(trial_index=trial_index, raw_data=result)
                active_jobs.remove((job, trial_index))
        while submitted_jobs < total_budget and len(active_jobs) < num_parallel_jobs:
            parameters_next, trial_index_next = ax_client.get_next_trial()

            job = slurm_queue_client.submit_training_job(parameters_next)
            print(f"Submitted as {job.job_id}")
            submitted_jobs += 1
            active_jobs.append((job, trial_index_next))
            sleep(1)

        exp_to_df(ax_client.experiment).to_csv(f"{EXPERIMENT_NAME}.csv")

        sleep(60)
    ax_client.save_to_json_file()


if __name__ == "__main__":
    main()
