# pml_vqvae

## When editing LaTex in VS Code

### Setup

1. Install [TexLive](https://www.tug.org/texlive/) (best would be to install the full version)
2. Install [Latex Workshop](https://marketplace.visualstudio.com/items?itemName=James-Yu.latex-workshop) VS Code extension
3. Make sure you have pulled the git version where there is a .vscode/settings.json file containing settings for latex workshop and a gitignore that has the file endings of auxiliary latex build files

### Workflow

#### TL;DR

Open the .tex file you want to edit, press Ctrl + Alt + B (Windows/Linux), Cmd + Option + B (macOS) to build the PDF and then Ctrl + Alt + V (Windows/Linux), Cmd + Option + V (macOS) to view the PDF. Then edit your file and you should get live updates in the PDF view.

#### Shortcuts

Generally, you should be able to open the .tex document that you are interested in and use the following commands to compile, view, and navigate the document:

1. Compile the LaTeX document
    - Shortcut: Ctrl + Alt + B (Windows/Linux), Cmd + Option + B (macOS)
    - Description: Compiles the main .tex file using the default recipe (e.g., latexmk).

1. View PDF Preview
    - Shortcut: Ctrl + Alt + V (Windows/Linux), Cmd + Option + V (macOS)
    - Description: Opens a PDF preview window beside the editor to view the compiled PDF. The preview updates automatically after each successful compilation.

1. Sync PDF with Source (Forward Sync)
    - Shortcut: Ctrl + Alt + J (Windows/Linux), Cmd + Option + J (macOS)
    - Description: Jumps from the .tex source in the editor to the corresponding location in the PDF preview.

1. Sync Source with PDF (Reverse Sync)
    - Shortcut: Ctrl + Click in the PDF Preview
    - Description: Jumps from a location in the PDF preview back to the corresponding place in the .tex source

## How to Log with Weights and Biases

1. You need to have a [Weights and Biases](https://wandb.ai/site/) (wandb) account
2. Add your wandb api key to your environment variables `export WANDB_API_KEY=<your key>`. If you run train the model within an apptainer make sure to add the environment variable to your `pml.def`-file
3. Create an experiment by adjusting the `config.yaml`-file. make sure `log_wandb=True`

## How to run

### Using the CLI

To run start the training, you can run the training script with
`python src/pml_vqvae/train.py`

On default (without extra parameters givin in the command), the training script will look up the `config.yaml`-file and will train according those defined parameters.

If you want to change the default values you can either change them in the `config.yaml` or you can pass the values yu want to change via the cli, e.g.

    python src/pml_vqvae/train.py --learning_rate 0.1

This will take the `config.yaml` as the foundation and overwrites all the given values passed in the command. See the `python src/pml_vqvae/train.py -h` for more information.

### Running on Cluster

1. ssh to cluster entry-node by `ssh <username>@hydra.ml.tu-berlin.de`
2. Build container if not exist, see Section [Build Container](#build-container)
3. Request a GPU node `srun --partition=gpu-test --gpus=1 --pty bash`
4. Go to project directory `cd pml_vqvae`
5. Run `./run_script.sh`

## Build Container
1. Obtain a node on the cpu only partition with enough run time `srun --partition=cpu-2h --pty bash`
2. Build the container with `apptainer build pml.sif pml.def` (can take up to 15min)
