import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
from pathlib import Path


class CelebAFolder(Dataset):
    def __init__(self, folder: str, transform):
        self.paths = sorted(Path(folder).glob("*.jpg"))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(image), 0


def get_transform(resolution: int = 256) -> T.Compose:
    return T.Compose([
        T.CenterCrop(178),
        T.Resize(resolution, antialias=True),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def get_dataloaders(config: dict) -> tuple[DataLoader, DataLoader]:
    transform = get_transform(config["resolution"])

    train_ds = CelebAFolder("data/train", transform)
    val_ds   = CelebAFolder("data/val",   transform)

    shared = dict(
        batch_size  = config["batch_size"],
        num_workers = config["num_workers"],
        pin_memory  = config["device"] == "cuda",
    )
    return (
        DataLoader(train_ds, shuffle=True,  **shared),
        DataLoader(val_ds,   shuffle=False, **shared),
    )
