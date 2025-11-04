import json

with open("./datasets/metadata_style_clustered_kmeans.json", "r") as f:
    data = json.load(f)

print("Number of samples:", len(data))
