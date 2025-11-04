from torch.utils.data import Dataset
from PIL import Image
import json, torch, os, numpy as np

class DeepfakeJSONDataset(Dataset):

    def __init__(self, json_path, clip_processor, style_extractor, transform=None):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.processor = clip_processor
        self.style_extractor = style_extractor
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        img_path = sample["path"]
        caption = sample["caption"].strip()
        label = 1.0 if sample["true_label"].lower() == "real" else 0.0

        img = Image.open(img_path).convert("RGB")
        if self.transform: img = self.transform(img)
        clip_inputs = self.processor(text=[caption], images=img, return_tensors="pt", padding=True)
        style_vec = torch.from_numpy(self.style_extractor(np.array(img))).float()

        sim = torch.tensor(sample.get("similarity", 0.0)).float()
        cluster = torch.tensor(sample.get("cluster_id_style", -1)).long()
        return clip_inputs, style_vec, torch.tensor(label).float(), sim, cluster
