import open_clip
import torch


class CLIPImageEncoder:
    def __init__(self, model_name: str = "ViT-B-16", device: str = "cuda") -> None:
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained="openai",
        )
        self.model = self.model.to(self.device)
        self.model.eval()

    def encode(self, image):
        # image expected as PIL image for open_clip preprocess
        image = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.model.encode_image(image)

        return feat
