#!/bin/bash

apptainer run --nv --bind /home/space/datasets:/home/space/datasets pml.sif python ${@:1}