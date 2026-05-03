import torch

from config import config
from dataset import get_dataloaders
from vqvae import VQ_VAE, vq_vae_loss


def check_vram():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    train_loader, _ = get_dataloaders(config)

    model     = VQ_VAE(config).to(config["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    images, _ = next(iter(train_loader))
    images = images.to(config["device"])

    x_recon, q_loss, _ = model(images)
    total_loss, _, _   = vq_vae_loss(x_recon, images, q_loss)
    total_loss.backward()

    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    print(f"Peak VRAM with batch_size={config['batch_size']}: {peak:.2f} GB")


if __name__ == "__main__":
    check_vram()
