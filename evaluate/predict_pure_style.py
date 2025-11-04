import torch
import sys
from PIL import Image
import numpy as np
from models.style_extractor_pure import PureStyleExtractor
import torch.nn as nn

class PureStyleClassifier(nn.Module):
    def __init__(self, style_dim=25, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(style_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    
    def forward(self, style_features):
        return self.net(style_features)

def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluate/predict_pure_style.py <image_path>")
        return
    
    image_path = sys.argv[1]
    checkpoint_path = "checkpoints/pure_style.pt"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    print(f"Loading model from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    style_dim = checkpoint.get("style_dim", 25)
    
    style_extractor = PureStyleExtractor(device)
    
    model = PureStyleClassifier(style_dim=style_dim).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    
    print(f"Style dimension: {style_dim} technical features")
    print(f"Architecture: 100% content-agnostic (NO CLIP)")
    
    print(f"\nAnalyzing image: {image_path}")
    print("-" * 50)
    
    img = Image.open(image_path).convert("RGB")
    
    style_vec = style_extractor(np.array(img))
    style_tensor = torch.tensor(style_vec, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(style_tensor)
        prob_real = torch.sigmoid(logits).item()
    
    prob_fake = 1 - prob_real
    prediction = "REAL" if prob_real > 0.5 else "FAKE"
    confidence = max(prob_real, prob_fake)
    
    print(f"\nPrediction: {prediction}")
    print(f"   Confidence: {confidence:.2%}")
    print(f"\nDetailed Probabilities:")
    print(f"   Real: {prob_real:.2%}")
    print(f"   Fake: {prob_fake:.2%}")
    print(f"\nDetection Method: Pure artifact analysis")
    print(f"  - Frequency domain patterns")
    print(f"  - Noise characteristics")
    print(f"  - Texture anomalies")
    print(f"  - Edge inconsistencies")
    print(f"  - Color correlations")
    print()

if __name__ == "__main__":
    main()

