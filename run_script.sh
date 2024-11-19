#!/bin/bash

apptainer run --nv --env-file .env --bind /home/space/datasets:/home/space/datasets pml.sif python ${@:1}