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


EXPERIMENT_NAME = "hyperopt-vqvae-ce-buffer-and-max-idle-count"


FIXED_HYPERPARAMS = {
    "dataset": "imagenet",
    "experiment_name": EXPERIMENT_NAME,
    "model_name": "vqvae-ce",
    "n_test": 5000,
    "n_train": 100000,
    "test_interval": 1,
    "vis_train_interval": 1,
    "label_conditioning": True,
    "epochs": 20,
    "optimizer": "adam",
    "wandb_log": True,
    "batch_size": 32,
    "learning_rate": 0.0001,
    "weight_decay": 0.0001,
    # PixelCNN params
    # "conditional": True,
    # "num_codes": 512,
    # "vqvae_path": "artifacts/konni_replacement_vqvae",
    # "dilations": "1-2-1-3-1-4-1-3-1-2-1",
    # "hidden_chan": 128,
    # VQVAE params
    "hidden_dimension": 128,
    "embedding_dimension": 128,
    "codebook_size": 256,
    "codebook_initialization_radius": 0.5,
    "commitment_weight": 4.0,
}


VQVAE_HYPERPARAMETER_SEARCH_SPACE = [
    # {
    #     "name": "hidden_dimension",
    #     "type": "choice",
    #     "values": [64, 128, 256],
    #     "sort_values": True,
    #     "is_ordered": True,
    # },
    # {
    #     "name": "embedding_dimension",
    #     "type": "choice",
    #     "values": [64, 128, 256, 512],
    #     "sort_values": True,
    #     "is_ordered": True,
    # },
    # {
    #     "name": "codebook_size",
    #     "type": "choice",
    #     "values": [64, 128, 256, 512, 1024],
    #     "sort_values": True,
    #     "is_ordered": True,
    # },
    # {
    #     "name": "codebook_initialization_radius",
    #     "type": "choice",
    #     "values": [0.01, 0.5, 1.0, 2.0, 10.0],
    #     "sort_values": True,
    #     "is_ordered": True,
    # },
    # {
    #     "name": "commitment_weight",
    #     "type": "choice",
    #     "values": [2.0, 4.0, 7.0, 10.0],
    #     "sort_values": True,
    #     "is_ordered": True,
    # },
    {
        "name": "buffer_size",
        "type": "choice",
        "values": [0, 5, 20, 100],
        "sort_values": True,
        "is_ordered": True,
    },
]

PIXELCNN_HYPERPARAMETER_SEARCH_SPACE = [
    {
        "name": "conditional_embedding_dim",
        "type": "choice",
        "values": [16, 64, 256, 1024],
        "sort_values": True,
        "is_ordered": True,
    },
]


TRAINING_HYPERPARAMETER_SEARCH_SPACE = [
    {
        "name": "dataset",
        "type": "choice",
        "values": [
            "latent artifacts/konni_replacement_vqvae/imagenet_latents_200000",
            "latent artifacts/konni_replacement_vqvae/imagenet_latents_20classes",
        ],
    },
    # {
    #     "name": "learning_rate",
    #     "type": "choice",
    #     "values": [1e-4, 1e-3],
    #     "sort_values": True,
    #     "is_ordered": True,
    # },
    # {
    #     "name": "batch_size",
    #     "type": "choice",
    #     "values": [32, 64, 128],
    #     "sort_values": True,
    #     "is_ordered": True,
    # },
    # {
    #     "name": "weight_decay",
    #     "type": "choice",
    #     "values": [1e-4, 1e-3, 1e-2, 1e-1],
    #     "sort_values": True,
    #     "is_ordered": True,
    # },
]


def test(parameters):
    train_config = {}
    model_config = {}

    parameters = parameters | FIXED_HYPERPARAMS

    for k, v in parameters.items():
        if k in get_annotations(TrainConfig).keys():
            train_config[k] = v
        else:
            model_config[k] = v

    if "20classes" in train_config["dataset"]:
        model_config["num_classes"] = 20
    else:
        model_config["num_classes"] = 1000

    train_config["model_config"] = model_config

    train_config = TrainConfig.from_dict(train_config)

    print(train_config)

    best_model_performance = pml_vqvae.train.train(train_config)

    return best_model_performance


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
            slurm_partition="gpu-teaching-2d",
            slurm_gpus_per_node=1,
            slurm_cpus_per_task=1,
            timeout_min=1200,
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
        + PIXELCNN_HYPERPARAMETER_SEARCH_SPACE,
        objectives={"mse": ObjectiveProperties(minimize=True)},
    )

    slurm_queue_client = SlurmJobQueueClient()

    total_budget = 50
    num_parallel_jobs = 2
    active_jobs = []
    submitted_jobs = 0

    while submitted_jobs < total_budget or active_jobs:
        for job, trial_index in active_jobs[:]:
            if job.done():
                try:
                    result = job.result()
                except:
                    ax_client.abandon_trial(trial_index=trial_index)
                    active_jobs.remove((job, trial_index))
                    continue
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
