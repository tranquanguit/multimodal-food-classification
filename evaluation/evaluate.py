import sys
from pathlib import Path

import torch
import yaml
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from captioning.generate_captions import CaptionGenerator
from datasets.food101_dataset import Food101Dataset
from embeddings.clip_image_encoder import CLIPImageEncoder
from embeddings.clip_text_encoder import CLIPTextEncoder
from models.multimodal_model import MultimodalClassifier


def load_config(config_path: str = "configs/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate(checkpoint_path: str = "checkpoints/multimodal_food101.pt"):
    config = load_config()
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    dataset = Food101Dataset("./data", split="test")
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    captioner = CaptionGenerator(device=device)
    img_encoder = CLIPImageEncoder(model_name=config["clip_model"], device=device)
    txt_encoder = CLIPTextEncoder(model_name=config["clip_model"], device=device)

    model = MultimodalClassifier(num_classes=config["num_classes"]).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for image, label in tqdm(loader, desc="Evaluating"):
            pil_image = to_pil_image(image.squeeze(0).cpu())

            img_feat = img_encoder.encode(pil_image)
            captions = captioner.generate(pil_image, num_sentences=int(config["captions_per_image"]))
            txt_feat = txt_encoder.encode(captions)

            logits = model(img_feat, txt_feat)
            pred = torch.argmax(logits, dim=1)

            all_preds.extend(pred.cpu().tolist())
            all_labels.extend(label.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    print(f"Accuracy: {acc:.4f}")


if __name__ == "__main__":
    evaluate()
