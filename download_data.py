from datasets import load_dataset

print("Downloading train split (20k)...")
ds_train = load_dataset("flwrlabs/celeba", split="train[:20000]")
ds_train.save_to_disk("data/celeba_train")
print("Done train")

print("Downloading val split (2k)...")
ds_val = load_dataset("flwrlabs/celeba", split="valid[:2000]")  # valid not validation
ds_val.save_to_disk("data/celeba_val")
print("Done val")
