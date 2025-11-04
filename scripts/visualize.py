import numpy as np
import torch
from sklearn.manifold import UMAP
import matplotlib.pyplot as plt
from models.fusion_model import FusionModel
from models.style_extractor import StyleExtractor
from utils.dataset import DeepfakeJSONDataset
from transformers import CLIPProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"

model = FusionModel().to(device)
model.load_state_dict(torch.load("checkpoints/best_model.pt"))
model.eval()

clip_proc = CLIPProcessor.from_pretrained("google/siglip-base-patch16-384")
style_extractor = StyleExtractor(device)
dataset = DeepfakeJSONDataset("data/fake_balanced_filtered/data.json", clip_proc, style_extractor)

embs, labels = [], []
with torch.no_grad():
    for i in range(len(dataset)):
        inputs, style, label, sim, cluster = dataset[i]
        pixel = inputs["pixel_values"].squeeze(0).to(device)
        ids = inputs["input_ids"].squeeze(0).to(device)
        mask = inputs["attention_mask"].squeeze(0).to(device)
        fused = model.clip.get_image_features(pixel)
        fused = torch.cat([
            fused.cpu(),
            model.clip.get_text_features(ids, mask).cpu(),
            style.unsqueeze(0)
        ], dim=1)
        embs.append(fused.squeeze().numpy())
        labels.append(label)

embs, labels = np.stack(embs), np.array(labels)
proj = UMAP(n_neighbors=15, min_dist=0.1).fit_transform(embs)
plt.scatter(proj[:,0], proj[:,1], c=labels, cmap='coolwarm', alpha=0.6)
plt.title("Style–Semantic Embedding Space")
plt.show()
