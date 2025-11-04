import json

# Check openfake_raw
with open("openfake-annotation/datasets/openfake_raw/metadata.json", "r") as f:
    data = json.load(f)
    labels = [s["true_label"] for s in data]
    print(f"openfake_raw:")
    print(f"  Total: {len(labels)}")
    print(f"  Real: {labels.count('real')}")
    print(f"  Fake: {labels.count('fake')}")

print()

# Check fake_balanced_filtered  
with open("openfake-annotation/datasets/fake_balanced_filtered/metadata.json", "r") as f:
    data = json.load(f)
    labels = [s["true_label"] for s in data]
    print(f"fake_balanced_filtered:")
    print(f"  Total: {len(labels)}")
    print(f"  Real: {labels.count('real')}")
    print(f"  Fake: {labels.count('fake')}")

