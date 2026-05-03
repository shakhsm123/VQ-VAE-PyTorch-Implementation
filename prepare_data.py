from datasets import load_from_disk
from pathlib import Path


def convert_split(src_path: str, dst_path: str, log_every: int = 1000):
    ds = load_from_disk(src_path)
    Path(dst_path).mkdir(parents=True, exist_ok=True)
    for i, sample in enumerate(ds):
        sample["image"].save(f"{dst_path}/{i:06d}.jpg")
        if i % log_every == 0:
            print(f"  {i}/{len(ds)}")
    print(f"Done — {len(ds)} images saved to {dst_path}")


if __name__ == "__main__":
    print("Converting train...")
    convert_split("data/celeba_train", "data/train", log_every=1000)

    print("Converting val...")
    convert_split("data/celeba_val", "data/val", log_every=500)
