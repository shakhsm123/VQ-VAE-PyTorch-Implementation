# train.py
import json
from pathlib import Path

import torch

from config import config
from dataset import get_dataloaders
from vqvae import VQ_VAE, vq_vae_loss


def run_epoch(model, dataloader, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss = total_recon = total_commit = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, _ in dataloader:
            images = images.to(device)
            x_recon, q_loss, _ = model(images)
            loss, recon_loss, commit_loss = vq_vae_loss(x_recon, images, q_loss)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss   += loss.item()
            total_recon  += recon_loss.item()
            total_commit += commit_loss.item()

    n = len(dataloader)
    return total_loss / n, total_recon / n, total_commit / n


def train(config: dict, out_dir: str = "checkpoints"):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    train_loader, val_loader = get_dataloaders(config)
    model     = VQ_VAE(config).to(config["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    start_epoch  = 0
    train_losses = []
    val_losses   = []

    # --- Resume if checkpoint exists ---
    ckpt_path = out_dir / "last.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=config["device"])
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch  = ckpt["epoch"] + 1
        train_losses = ckpt.get("train_losses", [])
        val_losses   = ckpt.get("val_losses", [])
        print(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, config["epochs"]):
        tr_loss, tr_recon, tr_commit = run_epoch(model, train_loader, optimizer, config["device"], train=True)
        vl_loss, vl_recon, vl_commit = run_epoch(model, val_loader,   optimizer, config["device"], train=False)

        utilization = (model.quantizer.usage_count > 0).float().mean().item()
        model.quantizer.usage_count.zero_()

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)

        print(
            f"Epoch {epoch+1}/{config['epochs']} | "
            f"train: {tr_loss:.4f} (recon: {tr_recon:.4f}, commit: {tr_commit:.4f}) | "
            f"val: {vl_loss:.4f} (recon: {vl_recon:.4f}, commit: {vl_commit:.4f}) | "
            f"codebook: {utilization*100:.1f}%"
        )

        # --- Checkpoint every epoch ---
        torch.save({
            "epoch":        epoch,
            "model":        model.state_dict(),
            "optimizer":    optimizer.state_dict(),
            "train_losses": train_losses,
            "val_losses":   val_losses,
        }, ckpt_path)

        # --- Save best val model separately ---
        if vl_loss <= min(val_losses):
            torch.save(model.state_dict(), out_dir / "best.pt")

    # --- Save loss history ---
    with open(out_dir / "losses.json", "w") as f:
        json.dump({"train": train_losses, "val": val_losses}, f, indent=2)

    return model, train_losses, val_losses


if __name__ == "__main__":
    train(config)
