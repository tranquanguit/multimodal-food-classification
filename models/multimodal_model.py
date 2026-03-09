import torch
import torch.nn as nn


class MultimodalClassifier(nn.Module):
    def __init__(self, embed_dim: int = 512, num_classes: int = 101) -> None:
        super().__init__()

        self.image_self_attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        self.text_self_attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)

        self.cross_img = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        self.cross_txt = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)

        self.classifier = nn.Linear(embed_dim * 2, num_classes)

    def forward(self, img_feat: torch.Tensor, txt_feat: torch.Tensor) -> torch.Tensor:
        img_feat = img_feat.unsqueeze(1) if img_feat.dim() == 2 else img_feat
        txt_feat = txt_feat.unsqueeze(1) if txt_feat.dim() == 2 else txt_feat

        img_out, _ = self.image_self_attn(img_feat, img_feat, img_feat)
        txt_out, _ = self.text_self_attn(txt_feat, txt_feat, txt_feat)

        cross_img, _ = self.cross_img(img_out, txt_out, txt_out)
        cross_txt, _ = self.cross_txt(txt_out, img_out, img_out)

        fusion = torch.cat([cross_img.squeeze(1), cross_txt.squeeze(1)], dim=1)
        logits = self.classifier(fusion)

        return logits
