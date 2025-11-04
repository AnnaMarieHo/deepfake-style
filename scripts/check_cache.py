import numpy as np

cache_path = "openfake-annotation/datasets/fake_balanced_filtered/cache/fused_embeddings.npz"
cache = np.load(cache_path)

print("Cache contents:")
for key in cache.files:
    print(f"  {key}: shape={cache[key].shape}, dtype={cache[key].dtype}")

labels = cache["label"]
print(f"\nLabel distribution:")
print(f"  Total samples: {len(labels)}")
print(f"  Real (label=1): {(labels == 1).sum()}")
print(f"  Fake (label=0): {(labels == 0).sum()}")
print(f"  Unique labels: {np.unique(labels)}")

if (labels == 1).sum() == 0:
    print("\nWARNING: No real samples found! Only fake samples in cache.")
    print("   You need to rerun preprocessing with both real AND fake samples.")

