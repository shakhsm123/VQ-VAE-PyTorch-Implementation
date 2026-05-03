import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("train.py loading...")
import json
print("json ok")
from pathlib import Path
print("pathlib ok")
import torch
print("torch ok")
from config import config
print("config ok")
from celeba_dataset import get_dataloaders
print("dataset ok")
from vqvae import VQ_VAE, vq_vae_loss
print("vqvae ok")


def run_epoch(model, dataloader, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss = total_recon = total_commit = 0

    print(f"  starting loop over {len(dataloader)} batches...")
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for i, (images, _) in enumerate(dataloader):
            if i == 0: print("  first batch loaded onto CPU")
            images = images.to(device)
            if i == 0: print("  first batch moved to", device)
            x_recon, q_loss, _ = model(images)
            if i == 0: print("  first forward pass ok")
            loss, recon_loss, commit_loss = vq_vae_loss(x_recon, images, q_loss)
            if i == 0: print("  first loss ok:", loss.item())

            if train:
                optimizer.zero_grad()
                loss.backward()
                if i == 0: print("  first backward ok")
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if i == 0: print("  first optimizer step ok")

            total_loss   += loss.item()
            total_recon  += recon_loss.item()
            total_commit += commit_loss.item()

            if i % 100 == 0:
                print(f"  batch {i}/{len(dataloader)} loss={loss.item():.4f}")

    n = len(dataloader)
    return total_loss / n, total_recon / n, total_commit / n


def train(config: dict, out_dir: str = "checkpoints"):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    print("creating model...")
    model = VQ_VAE(config).to(config["device"])
    print("model ok")
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    print("optimizer ok")

    start_epoch  = 0
    train_losses = []
    val_losses   = []

    ckpt_path = out_dir / "last.pt"
    if ckpt_path.exists():
        print("resuming from checkpoint...")
        ckpt = torch.load(ckpt_path, map_location=config["device"])
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch  = ckpt["epoch"] + 1
        train_losses = ckpt.get("train_losses", [])
        val_losses   = ckpt.get("val_losses", [])
        print(f"resumed from epoch {start_epoch}")

    print("loading dataloaders...")
    train_loader, val_loader = get_dataloaders(config)
    print(f"dataloaders ok — train: {len(train_loader)} batches, val: {len(val_loader)} batches")

    for epoch in range(start_epoch, config["epochs"]):
        print(f"\n=== Epoch {epoch+1}/{config['epochs']} ===")

        print("--- train ---")
        tr_loss, tr_recon, tr_commit = run_epoch(model, train_loader, optimizer, config["device"], train=True)
        print(f"train done: loss={tr_loss:.4f}")

        print("--- val ---")
        vl_loss, vl_recon, vl_commit = run_epoch(model, val_loader, optimizer, config["device"], train=False)
        print(f"val done: loss={vl_loss:.4f}")

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

        print("saving checkpoint...")
        torch.save({
            "epoch":        epoch,
            "model":        model.state_dict(),
            "optimizer":    optimizer.state_dict(),
            "train_losses": train_losses,
            "val_losses":   val_losses,
        }, ckpt_path)

        if vl_loss <= min(val_losses):
            torch.save(model.state_dict(), out_dir / "best.pt")
            print("saved best.pt")

    with open(out_dir / "losses.json", "w") as f:
        json.dump({"train": train_losses, "val": val_losses}, f, indent=2)

    return model, train_losses, val_losses


if __name__ == "__main__":
    print("calling train...")
    print("config device:", config["device"])
    train(config)


if __name__ == "__main__":
    train(config)
