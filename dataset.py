import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms as T


class CelebAHF(Dataset):
    def __init__(self, split: str, transform):
        self.ds = load_dataset("flwrlabs/celeba", split=split)
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        image = self.ds[idx]["image"]          # PIL image
        return self.transform(image), 0        # dummy label


def get_transform(resolution: int = 256) -> T.Compose:
    return T.Compose([
        T.CenterCrop(178),
        T.Resize(resolution, antialias=True),  # antialias suppresses warning
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def get_dataloaders(config: dict) -> tuple[DataLoader, DataLoader]:
    transform = get_transform(config["resolution"])

    train_ds = Subset(CelebAHF("train", transform), range(config["train_size"]))
    val_ds   = Subset(CelebAHF("valid", transform), range(config["val_size"]))

    shared = dict(
        batch_size  = config["batch_size"],
        num_workers = config["num_workers"],
        pin_memory  = config["device"] == "cuda",   # no-op on CPU, avoids warning
    )
    return (
        DataLoader(train_ds, shuffle=True,  **shared),
        DataLoader(val_ds,   shuffle=False, **shared),
    )