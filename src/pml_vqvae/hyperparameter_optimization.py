"""to run: apptainer run --env-file .env --nv --bind /home/space/,/etc/slurm,/opt/slurm,/opt/slurm-23.2,/etc/munge,
/var/run/munge,/usr/lib/x86_64-linux-gnu/libmunge.so.2 pml.sif python hyperparameter_optimization.py"""

import warnings
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.service.utils.report_utils import exp_to_df
import submitit
from time import sleep
import random
from pml_vqvae.train_config import TrainConfig
import pml_vqvae.train

warnings.simplefilter(action="ignore", category=FutureWarning)


def test(parameters):
    config_dict = {
        "model_name": "vqvae",
        "name": f"hyper_vqvae_{random.randint(1999999, 1000000)}",
        "seed": 42,
        "test_interval": None,
        "vis_train_interval": None,
        "wandb_log": True,
        "n_train": 500,
        "n_test": 100,
        "dataset": "imagenet",
        "epochs": 10,
        "class_idx": [],
        "hidden_dimensions": parameters["hidden_dimensions"],
        "codebook_size": parameters["codebook_size"],
        "beta_discrete_code_commitment": parameters["beta_discrete_code_commitment"],
        "optimizer": parameters["optimizer"],
        "learning_rate": parameters["learning_rate"],
        "momentum": parameters["momentum"],
        "weight_decay": parameters["weight_decay"],
        "batch_size": parameters["batch_size"],
    }
    config = TrainConfig.from_dict(config_dict)
    pml_vqvae.train.train(config)
    return parameters["learning_rate"] / parameters["beta_discrete_code_commitment"]


class SlurmJobQueueClient:
    def __init__(self):
        log_folder = "log_run/%j"
        running_dir = "/home/pml11/github_pml/"
        self.training_executor = submitit.AutoExecutor(
            folder=log_folder,
            cluster="slurm",
            slurm_python="/usr/bin/apptainer run --nv --env-file .env --bind "
            "/home/space/datasets:/home/space/datasets pml.sif python",
        )
        self.training_executor.update_parameters(
            slurm_partition="cpu-2h",
            # slurm_gpus_per_node=1,
            slurm_cpus_per_task=1,
            slurm_job_name="hyper_param_opt",
            slurm_additional_parameters={
                "chdir": running_dir,
            },
        )

    def submit_training_job(self, parameters):
        try:
            job = self.training_executor.submit(test, parameters)
        except:
            return False
        return job


def evaluate(parameters):
    return parameters["beta_discrete_code_commitment"] * parameters["learning_rate"]


def main():
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
                "values": ["adam", "sgd", "rmsprop"],
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
                "values": [32, 64, 128, 256, 512, 1024],
                "sort_values": True,
                "is_ordered": True,
            },
            {
                "name": "momentum",
                "type": "choice",
                "values": [0, 0.9],
                "sort_values": True,
                "is_ordered": True,
            },
            {
                "name": "weight_decay",
                "type": "range",
                "bounds": [1e-6, 1e-2],
                "log_scale": True,
            },
        ],
        objectives={"mse": ObjectiveProperties(minimize=True)},
    )

    slurm_queue_client = SlurmJobQueueClient()

    total_budget = 16
    num_parallel_jobs = 4
    jobs = []
    submitted_jobs = 0

    while submitted_jobs < total_budget or jobs:
        for job, trial_index in jobs[:]:
            if job.done():
                result = job.result()
                ax_client.complete_trial(trial_index=trial_index, raw_data=result)
                jobs.remove((job, trial_index))
        while submitted_jobs < total_budget and len(jobs) < num_parallel_jobs:
            parameters_next, trial_index_next = ax_client.get_next_trial()

            job = slurm_queue_client.submit_training_job(parameters_next)
            print(f"Submitted as {job.job_id}")
            submitted_jobs += 1
            jobs.append((job, trial_index_next))
            sleep(1)

        print(exp_to_df(ax_client.experiment))

        sleep(10)
    ax_client.save_to_json_file()


if __name__ == "__main__":
    main()
