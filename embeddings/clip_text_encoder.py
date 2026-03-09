import open_clip
import torch


class CLIPTextEncoder:
    def __init__(self, model_name: str = "ViT-B-16", device: str = "cuda") -> None:
        self.device = device
        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained="openai",
        )

        self.tokenizer = open_clip.get_tokenizer(model_name)

        self.model = self.model.to(self.device)
        self.model.eval()

    def encode(self, captions):
        tokens = self.tokenizer(captions).to(self.device)

        with torch.no_grad():
            feat = self.model.encode_text(tokens)

        feat = feat.mean(dim=0, keepdim=True)

        return feat
