"Python script to train different models"

import argparse

from pml_vqvae.baseline.vae import train_for_real

parser = argparse.ArgumentParser()
parser.add_argument("model", help="The model you want to train")
parser.add_argument("dataset",
                    help="The dataset onto which to train your model")

args = parser.parse_args()

if args.model == "vae" and args.dataset == "cifar10":
    train_for_real()
else:
    print("Until now only VAE on CIFAR10 supported.")
