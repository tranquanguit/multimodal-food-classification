import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
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
from utils.seed import set_seed


def load_config(config_path: str = "configs/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train():
    config = load_config()
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    set_seed(42)

    dataset = Food101Dataset("./data", split="train")
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    captioner = CaptionGenerator(device=device)
    img_encoder = CLIPImageEncoder(model_name=config["clip_model"], device=device)
    txt_encoder = CLIPTextEncoder(model_name=config["clip_model"], device=device)

    model = MultimodalClassifier(num_classes=config["num_classes"]).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["lr"]))

    epochs = int(config["epochs"])
    captions_per_image = int(config["captions_per_image"])

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for image, label in progress:
            pil_image = to_pil_image(image.squeeze(0).cpu())

            img_feat = img_encoder.encode(pil_image)
            captions = captioner.generate(pil_image, num_sentences=captions_per_image)
            txt_feat = txt_encoder.encode(captions)

            logits = model(img_feat, txt_feat)
            loss = criterion(logits, label.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(loader)
        print(f"Epoch {epoch + 1}: loss={epoch_loss:.4f}")

    Path("checkpoints").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/multimodal_food101.pt")
    print("Saved checkpoint to checkpoints/multimodal_food101.pt")


if __name__ == "__main__":
    train()
