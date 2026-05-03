import torch

config = {
    "resolution": 256,
    "batch_size": 16,
    "train_size": 20000,
    "val_size": 2000,
    "codebook_size": 1024,
    "codebook_dim": 256,
    "commitment_beta": 0.25,
    "ema_decay": 0.99,
    "dead_code_interval": 50,
    "lr": 3e-4,
    "epochs": 100,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 0,
}
