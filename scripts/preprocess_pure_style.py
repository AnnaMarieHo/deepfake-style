from PIL import Image
import os, json, numpy as np
from tqdm import tqdm
from models.style_extractor_pure import PureStyleExtractor


style_extractor = PureStyleExtractor("cpu")

in_json  = "openfake-annotation/datasets/combined/metadata.json"
out_path = "openfake-annotation/datasets/combined/cache/pure_style_embeddings.npz"

with open(in_json, "r", encoding="utf-8") as f:
    data = json.load(f)

style_embs, labels, clusters, sims = [], [], [], []

print(f"Processing {len(data)} samples...")
for sample in tqdm(data):
    img_path = sample["path"]
    if not os.path.exists(img_path):
        img_path = os.path.join("openfake-annotation", sample["path"])

    img = Image.open(img_path).convert("RGB")
    
    style_vec = style_extractor(np.array(img))[None, :]
    
    style_embs.append(style_vec)
    labels.append(1 if sample["true_label"] == "real" else 0)
    clusters.append(sample.get("cluster_id_style", -1))
    sims.append(sample.get("similarity", 0.0))

os.makedirs(os.path.dirname(out_path), exist_ok=True)
np.savez_compressed(
    out_path,
    style=np.vstack(style_embs),
    label=np.array(labels),
    cluster=np.array(clusters),
    similarity=np.array(sims),
)

print(f"\nSaved pure style embeddings to {out_path}")
print(f"  Style features: {style_embs[0].shape[1]} (25 technical features)")
print(f"  Total samples: {len(labels)}")
print(f"  Real: {sum(labels)}")
print(f"  Fake: {len(labels) - sum(labels)}")
print(f"\nFeatures:")
for i, name in enumerate(style_extractor.get_feature_names(), 1):
    print(f"  {i:2d}. {name}")

