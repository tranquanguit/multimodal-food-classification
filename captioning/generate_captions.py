import torch
from transformers import Blip2ForConditionalGeneration, Blip2Processor


class CaptionGenerator:
    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        ).to(self.device)

    def generate(self, image, num_sentences: int = 10):
        prompt = "Describe what this food image looks like and what ingredients it might contain."

        captions = []
        for _ in range(num_sentences):
            inputs = self.processor(image, text=prompt, return_tensors="pt").to(self.device)
            output = self.model.generate(**inputs, max_new_tokens=80)
            caption = self.processor.decode(output[0], skip_special_tokens=True)
            captions.append(caption)

        return captions
