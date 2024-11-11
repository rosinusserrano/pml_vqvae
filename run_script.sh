#!/bin/bash

apptainer run --nv --bind /home/space/datasets:/home/space/datasets --bind $HOME/pml_vqvae/artifacts:$HOME/pml_vqvae/artifacts pml.sif python ${@:1}