#!/bin/bash

apptainer run --nv --env-file .env \
    --bind /home/space/datasets:/home/space/datasets,/etc/slurm,/opt/slurm,/opt/slurm-23.2,/etc/munge,/var/run/munge,/usr/lib/x86_64-linux-gnu/libmunge.so.2 \
    pml.sif python ${@:1}