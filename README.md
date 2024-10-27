# pml_vqvae

## How to run on cluster?

1. ssh to cluster entry-node by `ssh <username>@hydra.ml.tu-berlin.de`
2. Build container if not exist, see Section [Build Conteiner](#build-container)
3. Request a GPU node `srun --partition=gpu-test --gpus=1 --pty bash`
4. Go to project directory `cd pml_vqvae`
5. Run `./run_script.sh`

## Build Container
1. Obtain a node on the cpu only partition with enough run time `srun --partition=cpu-2h --pty bash`
2. Build the container with `apptainer build pml.sif pml.def` (can take up to 15min)

