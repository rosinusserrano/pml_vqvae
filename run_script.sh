#!/bin/bash

apptainer run --nv --env-file .env \
    --bind /var/spool/slurmd:/var/spool/slurmd,/var/log/slurm:/var/log/slurm,/etc/passwd:/etc/passwd,/etc/group:/etc/group,/home/space/datasets:/home/space/datasets,/etc/slurm,/opt/slurm,/opt/slurm-23.2,/etc/munge,/var/run/munge,/usr/lib/x86_64-linux-gnu/libmunge.so.2 \
    pml.sif python ${@:1}