# config.py
import argparse
import torch


def get_config():
    parser = argparse.ArgumentParser(description="VQ-VAE training config")

    # Data
    parser.add_argument("--resolution",    type=int,   default=256)
    parser.add_argument("--batch_size",    type=int,   default=64)
    parser.add_argument("--train_size",    type=int,   default=20000)
    parser.add_argument("--val_size",      type=int,   default=2000)

    # Codebook
    parser.add_argument("--codebook_size", type=int,   default=1024)
    parser.add_argument("--codebook_dim",  type=int,   default=256)

    # Loss / EMA
    parser.add_argument("--commitment_beta", type=float, default=0.25)
    parser.add_argument("--ema_decay",       type=float, default=0.99)

    # Optimizer
    parser.add_argument("--lr",     type=float, default=3e-4)
    parser.add_argument("--epochs", type=int,   default=100)

    # System
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=2)

    args = parser.parse_args()
    return vars(args)  
config = {
    "resolution": 256,
    "batch_size": 64,
    "train_size": 20000,
    "val_size": 2000,
    "codebook_size": 1024,
    "codebook_dim": 256,
    "commitment_beta": 0.25,
    "ema_decay": 0.99,
    "lr": 3e-4,
    "epochs": 100,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 2,
}