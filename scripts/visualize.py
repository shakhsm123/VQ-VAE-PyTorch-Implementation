import argparse
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils

from config import config
from dataset import get_dataloaders
from vqvae import VQ_VAE


def visualize(checkpoint_path: str, n_images: int = 8):
    _, val_loader = get_dataloaders(config)

    model = VQ_VAE(config).to(config["device"])
    model.load_state_dict(torch.load(checkpoint_path, map_location=config["device"]))
    model.eval()

    with torch.no_grad():
        images, _ = next(iter(val_loader))
        images = images[:n_images].to(config["device"])
        x_recon, _, _ = model(images)

    # Denormalize [-1, 1] → [0, 1]
    images  = (images  * 0.5 + 0.5).clamp(0, 1).cpu()
    x_recon = (x_recon * 0.5 + 0.5).clamp(0, 1).cpu()

    # Reconstruction grid
    orig_grid  = vutils.make_grid(images,  nrow=n_images)
    recon_grid = vutils.make_grid(x_recon, nrow=n_images)

    fig, axes = plt.subplots(2, 1, figsize=(16, 4))
    axes[0].imshow(orig_grid.permute(1, 2, 0))
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(recon_grid.permute(1, 2, 0))
    axes[1].set_title("Reconstruction")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig("reconstructions.png", dpi=150)
    plt.show()

    # Codebook usage histogram
    usage = model.quantizer.usage_count.cpu().numpy()
    plt.figure(figsize=(12, 3))
    plt.bar(range(len(usage)), usage)
    plt.title("Codebook usage")
    plt.xlabel("Code index")
    plt.ylabel("Usage count")
    plt.tight_layout()
    plt.savefig("codebook_usage.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--n_images",   type=int, default=8)
    args = parser.parse_args()
    visualize(args.checkpoint, args.n_images)
