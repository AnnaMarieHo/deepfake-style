import os, json, argparse, torch, gc, time
from tqdm import tqdm
from PIL import Image, ImageFile, UnidentifiedImageError
from transformers import AutoProcessor, AutoModel
from datasets import load_dataset
import pandas as pd

ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="ComplexDataLab/OpenFake")
parser.add_argument("--split", type=str, default="train")
parser.add_argument("--threshold", type=float, default=0.08)
parser.add_argument("--target_count", type=int, default=9966)
parser.add_argument("--out_dir", type=str, default="datasets/fake_balanced_filtered")
parser.add_argument("--save_every", type=int, default=25, help="Save metadata every N images")
args = parser.parse_args()

out_img_dir = os.path.join(args.out_dir, "images")
json_path = os.path.join(args.out_dir, "metadata.json")


os.makedirs(args.out_dir)
os.makedirs(out_img_dir, exist_ok=True)

records, processed_paths = [], set()
saved, tried = 0, 0

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "google/siglip-base-patch16-512"
print(f"Loading {model_id} on {device} ...")
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id).to(device).eval()

print(f"Streaming {args.dataset_name} ({args.split}) from beginning...")
dataset = load_dataset(args.dataset_name, split=args.split, streaming=True)

def save_metadata(force=False):
    if force or (saved and saved % args.save_every == 0):
        tmp_path = json_path + ".tmp"
        try:
            with open(tmp_path, "w") as f:
                json.dump(records, f, indent=2)
            os.replace(tmp_path, json_path)
            print(f"Saved progress — {len(records)} entries written.")
        except Exception as e:
            print(f"Metadata save failed: {e}")
        torch.cuda.empty_cache(); gc.collect()

start_time = time.time()

for sample in tqdm(dataset, desc="Streaming fake samples"):
    label = str(sample.get("label", "")).lower()
    if label != "fake":
        continue
    tried += 1
    try:
        img = sample["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
    except (UnidentifiedImageError, OSError) as e:
        print(f"Skipping corrupt image at index {tried}: {e}")
        continue

    caption = str(sample.get("prompt") or sample.get("caption") or "")

    try:
        inputs = processor(text=[caption], images=img, return_tensors="pt",
                           padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            sim = float((outputs.image_embeds @ outputs.text_embeds.T).item())
        del inputs, outputs
    except torch.cuda.OutOfMemoryError:
        print("CUDA OOM, retrying after cache clear.")
        torch.cuda.empty_cache(); time.sleep(3)
        continue
    except Exception as e:
        print(f"SigLIP failed on sample {tried}: {e}")
        torch.cuda.empty_cache()
        continue

    if sim >= args.threshold:
        try:
            filename = f"{saved:05d}.jpg"
            path = os.path.join(out_img_dir, filename)
            img.save(path)
            records.append({
                "path": path,
                "caption": caption,
                "true_label": "fake",
                "similarity": sim
            })
            processed_paths.add(path)
            saved += 1
            print(f"[{saved}/{args.target_count}] Saved {filename} | sim={sim:.3f}")
            print(f"   Caption: {caption[:120]}{'...' if len(caption) > 120 else ''}\n")
            save_metadata()
        except Exception as e:
            print(f"Save error at sample {tried}: {e}")

    if saved and saved % 100 == 0:
        elapsed = time.time() - start_time
        rate = tried / max(1, elapsed)
        pass_rate = saved / max(1, tried)
        est_needed = int((args.target_count - saved) / max(1e-9, pass_rate))
        print(f"[proj] pass_rate={pass_rate:.2%}, throughput={rate:.2f}/s, "
              f"remaining≈{est_needed:,} fakes (~{est_needed/rate/3600:.1f}h)")

    if saved >= args.target_count:
        print(f"\n Reached target of {args.target_count} high-similarity fakes.")
        break

save_metadata(force=True)
print(f"Final metadata written ({len(records)} samples).")
print(f"Metadata path: {json_path}")
